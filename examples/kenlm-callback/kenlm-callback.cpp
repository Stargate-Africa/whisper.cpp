#include "common.h"
#include "common-whisper.h"

#include "whisper.h"

#include <lm/model.hh>
#include <lm/virtual_interface.hh>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct params {
    std::string model_path;
    std::string audio_path;
    std::string kenlm_path;
    std::string language = "en";
    std::vector<std::string> hotwords;
    int threads = std::max(1u, std::thread::hardware_concurrency() > 4 ? 4u : std::thread::hardware_concurrency());
    int beam_size = 5;
    int rescore_top_k = 5;
    float lm_alpha = 8.0f;
    float hotword_bias = 3.0f;
    bool no_gpu = true;
};

struct hotword_entry {
    std::string phrase;
    std::vector<whisper_token> tokens;
};

struct kenlm_callback_data {
    std::unique_ptr<lm::base::Model> model;
    std::vector<hotword_entry> hotwords;
    int rescore_top_k;
    float lm_alpha;
    float hotword_bias;

    explicit kenlm_callback_data(const std::string & model_path, int top_k, float alpha, float bias)
        : model(lm::ngram::LoadVirtual(model_path.c_str())), rescore_top_k(top_k), lm_alpha(alpha), hotword_bias(bias) {
    }
};

static void print_usage(const char * argv0) {
    std::cerr
    << "usage: " << argv0 << " --model MODEL --audio AUDIO --kenlm-model LM [options]\n"
        << "\n"
        << "options:\n"
        << "  --model PATH            whisper ggml model path\n"
        << "  --audio PATH            input audio path\n"
        << "  --kenlm-model PATH      KenLM binary or ARPA model path\n"
    << "  --hotword PHRASE        optional hotword or phrase to bias (repeatable)\n"
        << "  --language LANG         spoken language code (default: en)\n"
        << "  --beam-size N           beam size (default: 5)\n"
    << "  --rescore-top-k N       number of next-token candidates to rescore with KenLM on each step (default: 5, 0 disables general rescoring)\n"
        << "  --threads N             decode threads (default: min(4, hw))\n"
        << "  --lm-alpha N            KenLM score multiplier (default: 8.0)\n"
        << "  --hotword-bias N        base additive bias for hotword continuations (default: 3.0)\n"
        << "  --gpu                   enable GPU instead of CPU-only mode\n";
}

static std::string lowercase_ascii(std::string text) {
    for (size_t i = 0; i < text.size(); ++i) {
        text[i] = (char) std::tolower((unsigned char) text[i]);
    }
    return text;
}

static std::string resolve_whisper_language(const std::string & requested_language) {
    const std::string normalized = lowercase_ascii(requested_language);
    if (normalized.empty() || normalized == "auto") {
        return "auto";
    }

    if (whisper_lang_id(normalized.c_str()) >= 0) {
        return normalized;
    }

    return "auto";
}

static std::string normalize_for_kenlm(const std::string & text) {
    std::string out;
    out.reserve(text.size());

    bool prev_space = true;
    for (size_t i = 0; i < text.size(); ++i) {
        const unsigned char ch = (unsigned char) text[i];
        if (std::isalnum(ch) || ch == '\'' || ch == '-') {
            out.push_back((char) std::tolower(ch));
            prev_space = false;
        } else if (!prev_space) {
            out.push_back(' ');
            prev_space = true;
        }
    }

    if (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }

    return out;
}

static std::vector<std::string> split_words(const std::string & text) {
    std::istringstream stream(text);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

static double score_text(kenlm_callback_data & data, const std::string & text, bool eos = false) {
    const std::string normalized = normalize_for_kenlm(text);
    const auto words = split_words(normalized);
    if (words.empty() || !data.model) {
        return 0.0;
    }

    std::vector<char> state(data.model->StateSize());
    std::vector<char> out_state(data.model->StateSize());
    double total = 0.0;
    const auto & vocab = data.model->BaseVocabulary();

    data.model->BeginSentenceWrite(state.data());

    for (size_t i = 0; i < words.size(); ++i) {
        const auto word_index = vocab.Index(words[i]);
        total += data.model->BaseFullScore(state.data(), word_index, out_state.data()).prob;
        state.swap(out_state);
    }

    if (eos) {
        total += data.model->BaseFullScore(state.data(), vocab.EndSentence(), out_state.data()).prob;
    }

    return total;
}

static std::vector<whisper_token> tokenize_text(struct whisper_context * ctx, const std::string & text) {
    std::vector<whisper_token> tokens(std::max<int>(16, (int) text.size() * 2));
    int n_tokens = whisper_tokenize(ctx, text.c_str(), tokens.data(), (int) tokens.size());
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = whisper_tokenize(ctx, text.c_str(), tokens.data(), (int) tokens.size());
    }
    if (n_tokens <= 0) {
        return {};
    }
    tokens.resize(n_tokens);
    return tokens;
}

static std::string decode_tokens_text(
        struct whisper_context * ctx,
        const whisper_token_data * tokens,
        int n_tokens) {
    std::string result;
    for (int i = 0; i < n_tokens; ++i) {
        if (tokens[i].id >= whisper_token_eot(ctx)) {
            continue;
        }
        result += whisper_token_to_str(ctx, tokens[i].id);
    }
    return result;
}

static bool token_piece_is_boundary_candidate(const std::string & piece) {
    if (piece.empty()) {
        return false;
    }

    const unsigned char first = (unsigned char) piece.front();
    if (std::isspace(first)) {
        return true;
    }

    bool has_alnum = false;
    bool has_punct = false;
    for (size_t i = 0; i < piece.size(); ++i) {
        const unsigned char ch = (unsigned char) piece[i];
        has_alnum = has_alnum || std::isalnum(ch);
        has_punct = has_punct || std::ispunct(ch);
    }

    // Pure punctuation tokens are reasonable boundary cues, but mid-word
    // subword fragments should not trigger global KenLM rescoring.
    return has_punct && !has_alnum;
}

static int matched_prefix_len(const std::vector<whisper_token> & current, const std::vector<whisper_token> & target) {
    if (current.empty() || target.empty()) {
        return 0;
    }

    const int max_prefix = std::min<int>((int) current.size(), (int) target.size() - 1);
    for (int prefix = max_prefix; prefix >= 1; --prefix) {
        bool equal = true;
        for (int i = 0; i < prefix; ++i) {
            if (current[(int) current.size() - prefix + i] != target[i]) {
                equal = false;
                break;
            }
        }
        if (equal) {
            return prefix;
        }
    }

    return 0;
}

static void kenlm_hotword_callback(
        struct whisper_context * ctx,
          struct whisper_state * /* state */,
      const whisper_token_data * tokens,
                           int   n_tokens,
                         float * logits,
                          void * user_data) {
    if (ctx == nullptr || logits == nullptr || user_data == nullptr) {
        return;
    }

    auto * data = static_cast<kenlm_callback_data *>(user_data);
    const std::string current_text = decode_tokens_text(ctx, tokens, n_tokens);
    const double base_score = score_text(*data, current_text, false);
    const int n_vocab = whisper_n_vocab(ctx);

    if (data->rescore_top_k > 0 && n_vocab > 0) {
        std::vector<std::pair<float, whisper_token>> ranked;
        ranked.reserve(n_vocab);
        for (int id = 0; id < n_vocab; ++id) {
            if (id >= whisper_token_eot(ctx)) {
                continue;
            }
            if (!std::isfinite(logits[id])) {
                continue;
            }
            ranked.emplace_back(logits[id], id);
        }

        const int k = std::min<int>(data->rescore_top_k, (int) ranked.size());
        if (k > 0) {
            std::partial_sort(
                ranked.begin(),
                ranked.begin() + k,
                ranked.end(),
                [](const std::pair<float, whisper_token> & a, const std::pair<float, whisper_token> & b) {
                    return a.first > b.first;
                });

            for (int i = 0; i < k; ++i) {
                const whisper_token candidate_id = ranked[i].second;
                const std::string next_piece = whisper_token_to_str(ctx, candidate_id);
                if (!current_text.empty() && !token_piece_is_boundary_candidate(next_piece)) {
                    continue;
                }
                const std::string candidate_text = current_text + next_piece;
                const double candidate_score = score_text(*data, candidate_text, false);
                const float delta = (float) (candidate_score - base_score);
                logits[candidate_id] += data->lm_alpha * delta;
            }
        }
    }

    std::vector<whisper_token> current_ids;
    current_ids.reserve(std::max(0, n_tokens));
    for (int i = 0; i < n_tokens; ++i) {
        current_ids.push_back(tokens[i].id);
    }

    for (size_t i = 0; i < data->hotwords.size(); ++i) {
        const auto & hotword = data->hotwords[i];
        if (hotword.tokens.empty()) {
            continue;
        }

        const int matched = matched_prefix_len(current_ids, hotword.tokens);
        const whisper_token next_token = hotword.tokens[matched];
        const std::string next_piece = whisper_token_to_str(ctx, next_token);
        const std::string candidate_text = current_text + next_piece;
        const double candidate_score = score_text(*data, candidate_text, false);
        const float delta = (float) (candidate_score - base_score);
        logits[next_token] += data->hotword_bias + data->lm_alpha * delta;
    }
}

static bool parse_args(int argc, char ** argv, params & out) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (arg == "--model") {
            out.model_path = next("--model");
        } else if (arg == "--audio") {
            out.audio_path = next("--audio");
        } else if (arg == "--kenlm-model") {
            out.kenlm_path = next("--kenlm-model");
        } else if (arg == "--hotword") {
            out.hotwords.push_back(next("--hotword"));
        } else if (arg == "--language") {
            out.language = next("--language");
        } else if (arg == "--beam-size") {
            out.beam_size = std::max(1, std::atoi(next("--beam-size").c_str()));
        } else if (arg == "--rescore-top-k") {
            out.rescore_top_k = std::max(0, std::atoi(next("--rescore-top-k").c_str()));
        } else if (arg == "--threads") {
            out.threads = std::max(1, std::atoi(next("--threads").c_str()));
        } else if (arg == "--lm-alpha") {
            out.lm_alpha = std::strtof(next("--lm-alpha").c_str(), nullptr);
        } else if (arg == "--hotword-bias") {
            out.hotword_bias = std::strtof(next("--hotword-bias").c_str(), nullptr);
        } else if (arg == "--gpu") {
            out.no_gpu = false;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    return !out.model_path.empty() && !out.audio_path.empty() && !out.kenlm_path.empty() && (out.rescore_top_k > 0 || !out.hotwords.empty());
}

int main(int argc, char ** argv) {
    params p;
    try {
        if (!parse_args(argc, argv, p)) {
            print_usage(argv[0]);
            return 1;
        }
    } catch (const std::exception & exc) {
        std::cerr << "error: " << exc.what() << "\n\n";
        print_usage(argv[0]);
        return 1;
    }

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = !p.no_gpu;
    cparams.flash_attn = true;
    cparams.gpu_device = 0;

    std::unique_ptr<whisper_context, decltype(&whisper_free)> ctx(
        whisper_init_from_file_with_params(p.model_path.c_str(), cparams),
        whisper_free);
    if (!ctx) {
        std::cerr << "error: failed to load whisper model\n";
        return 2;
    }

    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(p.audio_path, pcmf32, pcmf32s, false)) {
        std::cerr << "error: failed to read audio file\n";
        return 3;
    }

    kenlm_callback_data callback_data(p.kenlm_path, p.rescore_top_k, p.lm_alpha, p.hotword_bias);
    for (size_t i = 0; i < p.hotwords.size(); ++i) {
        hotword_entry entry;
        entry.phrase = p.hotwords[i];
        entry.tokens = tokenize_text(ctx.get(), " " + entry.phrase);
        if (!entry.tokens.empty()) {
            callback_data.hotwords.push_back(entry);
        }
    }

    if (callback_data.rescore_top_k <= 0 && callback_data.hotwords.empty()) {
        std::cerr << "error: enable --rescore-top-k or provide at least one tokenizable --hotword\n";
        return 4;
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
    wparams.n_threads = p.threads;
    wparams.no_timestamps = true;
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;
    const std::string whisper_language = resolve_whisper_language(p.language);
    if (whisper_language != lowercase_ascii(p.language)) {
        std::cerr << "whisper.cpp does not recognize language code '" << p.language
                  << "'; falling back to auto-detect\n";
    }

    wparams.language = whisper_language.c_str();
    wparams.beam_search.beam_size = p.beam_size;
    wparams.logits_filter_callback = kenlm_hotword_callback;
    wparams.logits_filter_callback_user_data = &callback_data;

    std::cerr << "Loaded KenLM model: " << p.kenlm_path << "\n";
    std::cerr << "Rescore top-k: " << callback_data.rescore_top_k
              << ", hotword count: " << callback_data.hotwords.size()
              << ", lm_alpha=" << p.lm_alpha
              << ", hotword_bias=" << p.hotword_bias << "\n";

    if (whisper_full(ctx.get(), wparams, pcmf32.data(), (int) pcmf32.size()) != 0) {
        std::cerr << "error: whisper_full failed\n";
        return 5;
    }

    const int n_segments = whisper_full_n_segments(ctx.get());
    for (int i = 0; i < n_segments; ++i) {
        std::cout << whisper_full_get_segment_text(ctx.get(), i);
    }
    std::cout << std::endl;

    return 0;
}
