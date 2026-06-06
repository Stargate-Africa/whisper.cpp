set(TARGET whisper-kenlm-callback)
add_executable(${TARGET} kenlm-callback.cpp)

include(DefaultTargetOptions)

unset(KENLM_CALLBACK_EXTRA_LIBS)

find_package(ZLIB)
if (ZLIB_FOUND)
	list(APPEND KENLM_CALLBACK_EXTRA_LIBS ${ZLIB_LIBRARIES})
endif()

find_package(BZip2)
if (BZIP2_FOUND)
	list(APPEND KENLM_CALLBACK_EXTRA_LIBS ${BZIP2_LIBRARIES})
endif()

find_package(LibLZMA)
if (LIBLZMA_FOUND)
	list(APPEND KENLM_CALLBACK_EXTRA_LIBS ${LIBLZMA_LIBRARIES})
endif()

target_include_directories(${TARGET} PRIVATE ${KENLM_INCLUDE_DIR})
target_compile_definitions(${TARGET} PRIVATE KENLM_MAX_ORDER=${WHISPER_KENLM_MAX_ORDER})
target_link_libraries(${TARGET} PRIVATE common whisper ${KENLM_LIBRARY} ${KENLM_UTIL_LIBRARY} ${KENLM_CALLBACK_EXTRA_LIBS} ${FFMPEG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} RUNTIME)
