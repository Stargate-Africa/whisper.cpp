#include <whisper.h>

#include <lm/model.hh>
#include <lm/virtual_interface.hh>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace {

struct hotword_entry {
    std::string phrase;
    std::vector<whisper_token> tokens;
};

struct callback_data {
    std::shared_ptr<lm::base::Model> model;
    std::vector<hotword_entry> hotwords;
    int rescore_top_k = 0;
    float lm_alpha = 0.0f;
    float hotword_bias = 0.0f;
};

std::string lowercase_ascii(std::string text) {
    for (size_t i = 0; i < text.size(); ++i) {
        text[i] = (char) std::tolower((unsigned char) text[i]);
    }
    return text;
}

std::string normalize_for_kenlm(const std::string & text) {
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

std::vector<std::string> split_words(const std::string & text) {
    std::istringstream stream(text);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

double score_text(callback_data & data, const std::string & text, bool eos = false) {
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

std::string decode_tokens_text(
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

bool token_piece_is_boundary_candidate(const std::string & piece) {
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

    return has_punct && !has_alnum;
}

std::vector<whisper_token> tokenize_text(struct whisper_context * ctx, const std::string & text) {
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

int matched_prefix_len(const std::vector<whisper_token> & current, const std::vector<whisper_token> & target) {
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

std::string resolve_whisper_language(const std::string & requested_language) {
    const std::string normalized = lowercase_ascii(requested_language);
    if (normalized.empty() || normalized == "auto") {
        return "auto";
    }
    if (whisper_lang_id(normalized.c_str()) >= 0) {
        return normalized;
    }
    return "auto";
}

std::shared_ptr<lm::base::Model> load_kenlm_model_cached(const std::string & path) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::weak_ptr<lm::base::Model>> cache;

    std::lock_guard<std::mutex> guard(cache_mutex);
    auto it = cache.find(path);
    if (it != cache.end()) {
        auto locked = it->second.lock();
        if (locked) {
            return locked;
        }
    }

    std::shared_ptr<lm::base::Model> model(lm::ngram::LoadVirtual(path.c_str()));
    cache[path] = model;
    return model;
}

void kenlm_rescore_callback(
        struct whisper_context * ctx,
          struct whisper_state * /* state */,
      const whisper_token_data * tokens,
                           int   n_tokens,
                         float * logits,
                          void * user_data) {
    if (ctx == nullptr || logits == nullptr || user_data == nullptr) {
        return;
    }

    auto * data = static_cast<callback_data *>(user_data);
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

class WhisperCppBinding {
  public:
    WhisperCppBinding(const std::string & model_path, bool use_gpu = false, int gpu_device = 0) {
        whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = use_gpu;
        cparams.flash_attn = true;
        cparams.gpu_device = gpu_device;

        ctx_ = whisper_init_from_file_with_params(model_path.c_str(), cparams);
        if (ctx_ == nullptr) {
            throw std::runtime_error("failed to initialize whisper context");
        }
    }

    ~WhisperCppBinding() {
        if (ctx_ != nullptr) {
            whisper_free(ctx_);
            ctx_ = nullptr;
        }
    }

    py::dict transcribe_samples(
            py::array_t<float, py::array::c_style | py::array::forcecast> samples,
            const std::string & language = "auto",
            const std::string & kenlm_model_path = "",
            int beam_size = 5,
            int rescore_top_k = 0,
            float lm_alpha = 0.25f,
            float hotword_bias = 0.0f,
            std::vector<std::string> hotwords = {},
            bool no_timestamps = true,
            bool single_segment = false,
            int n_threads = 4) {
        if (ctx_ == nullptr) {
            throw std::runtime_error("whisper context is not initialized");
        }

        auto view = samples.unchecked<1>();
        if (view.shape(0) <= 0) {
            py::dict empty;
            empty["text"] = "";
            empty["language"] = language.empty() ? "auto" : language;
            empty["segments"] = py::list();
            return empty;
        }

        whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
        params.n_threads = std::max(1, n_threads);
        params.no_timestamps = no_timestamps;
        params.single_segment = single_segment;
        params.print_progress = false;
        params.print_realtime = false;
        params.print_timestamps = !no_timestamps;
        params.beam_search.beam_size = std::max(1, beam_size);

        std::string resolved_language = resolve_whisper_language(language);
        params.language = resolved_language.c_str();

        std::vector<float> pcm(view.size());
        for (ssize_t i = 0; i < view.shape(0); ++i) {
            pcm[(size_t) i] = view(i);
        }

        callback_data cb;
        std::string normalized_lm_path = kenlm_model_path;
        if (!normalized_lm_path.empty()) {
            cb.model = load_kenlm_model_cached(normalized_lm_path);
            cb.rescore_top_k = std::max(0, rescore_top_k);
            cb.lm_alpha = lm_alpha;
            cb.hotword_bias = hotword_bias;
            for (const auto & hotword : hotwords) {
                hotword_entry entry;
                entry.phrase = hotword;
                entry.tokens = tokenize_text(ctx_, " " + hotword);
                if (!entry.tokens.empty()) {
                    cb.hotwords.push_back(std::move(entry));
                }
            }
            params.logits_filter_callback = kenlm_rescore_callback;
            params.logits_filter_callback_user_data = &cb;
        }

        py::gil_scoped_release release;
        const int rc = whisper_full(ctx_, params, pcm.data(), (int) pcm.size());
        py::gil_scoped_acquire acquire;
        if (rc != 0) {
            throw std::runtime_error("whisper_full failed with code " + std::to_string(rc));
        }

        const int n_segments = whisper_full_n_segments(ctx_);
        py::list segments;
        std::string full_text;
        for (int i = 0; i < n_segments; ++i) {
            const char * text = whisper_full_get_segment_text(ctx_, i);
            const double t0 = whisper_full_get_segment_t0(ctx_, i) / 100.0;
            const double t1 = whisper_full_get_segment_t1(ctx_, i) / 100.0;
            std::string segment_text = text ? std::string(text) : std::string();
            full_text += segment_text;
            py::dict seg;
            seg["text"] = segment_text;
            seg["start"] = t0;
            seg["end"] = t1;
            segments.append(std::move(seg));
        }

        py::dict result;
        result["text"] = full_text;
        result["language"] = resolved_language;
        result["segments"] = segments;
        return result;
    }

  private:
    struct whisper_context * ctx_ = nullptr;
};

} // namespace

PYBIND11_MODULE(_whispercpp_binding, m) {
    m.doc() = "In-process whisper.cpp binding with optional KenLM callback rescoring";

    py::class_<WhisperCppBinding>(m, "WhisperCppBinding")
        .def(py::init<const std::string &, bool, int>(),
             py::arg("model_path"),
             py::arg("use_gpu") = false,
             py::arg("gpu_device") = 0)
        .def("transcribe_samples",
             &WhisperCppBinding::transcribe_samples,
             py::arg("samples"),
             py::arg("language") = "auto",
             py::arg("kenlm_model_path") = "",
             py::arg("beam_size") = 5,
             py::arg("rescore_top_k") = 0,
             py::arg("lm_alpha") = 0.25f,
             py::arg("hotword_bias") = 0.0f,
             py::arg("hotwords") = std::vector<std::string>{},
             py::arg("no_timestamps") = true,
             py::arg("single_segment") = false,
             py::arg("n_threads") = 4);
}
