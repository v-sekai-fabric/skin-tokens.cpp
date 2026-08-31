include_guard(GLOBAL)

function(skintokens_prepare_ggml output_variable)
  cmake_parse_arguments(PREPARE "" "SOURCE_DIR;PATCH_DIR;OUTPUT_DIR;EXPECTED_REVISION" "" ${ARGN})
  foreach(required SOURCE_DIR PATCH_DIR OUTPUT_DIR EXPECTED_REVISION)
    if(NOT PREPARE_${required})
      message(FATAL_ERROR "skintokens_prepare_ggml requires ${required}")
    endif()
  endforeach()
  if(NOT EXISTS "${PREPARE_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "GGML source is missing CMakeLists.txt: ${PREPARE_SOURCE_DIR}")
  endif()

  file(GLOB patch_files CONFIGURE_DEPENDS LIST_DIRECTORIES false "${PREPARE_PATCH_DIR}/*.patch")
  list(SORT patch_files)
  if(NOT patch_files)
    message(FATAL_ERROR "No GGML patches found in ${PREPARE_PATCH_DIR}")
  endif()

  find_package(Git 2.20 REQUIRED)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${PREPARE_SOURCE_DIR}" rev-parse HEAD
    RESULT_VARIABLE revision_result
    OUTPUT_VARIABLE source_revision
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(revision_result EQUAL 0)
    if(NOT source_revision STREQUAL PREPARE_EXPECTED_REVISION)
      message(FATAL_ERROR
        "GGML is at ${source_revision}, but the bundled patches require "
        "${PREPARE_EXPECTED_REVISION}. Update the git submodule or explicitly "
        "disable SKINTOKENS_APPLY_GGML_PATCHES for an already-patched source.")
    endif()
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${PREPARE_SOURCE_DIR}" status --porcelain --untracked-files=all
      RESULT_VARIABLE status_result
      OUTPUT_VARIABLE source_status
      ERROR_VARIABLE status_error
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT status_result EQUAL 0)
      message(FATAL_ERROR "Could not inspect GGML source: ${status_error}")
    endif()
    if(source_status)
      message(FATAL_ERROR
        "The GGML submodule must be pristine before patches are applied. "
        "Local changes found:\n${source_status}")
    endif()
  else()
    # Release source archives may contain the populated submodule without its
    # Git metadata. Patch applicability below remains the authoritative check.
    set(source_revision "archive-${PREPARE_EXPECTED_REVISION}")
  endif()

  set(fingerprint_material "prepare-ggml-v2\n${source_revision}\n")
  foreach(patch IN LISTS patch_files)
    file(SHA256 "${patch}" patch_sha256)
    string(APPEND fingerprint_material "${patch}:${patch_sha256}\n")
  endforeach()
  string(SHA256 fingerprint "${fingerprint_material}")
  set(stamp "${PREPARE_OUTPUT_DIR}/.skintokens-patch-stamp")
  set(rebuild TRUE)
  if(EXISTS "${stamp}" AND EXISTS "${PREPARE_OUTPUT_DIR}/CMakeLists.txt")
    file(READ "${stamp}" existing_fingerprint)
    string(STRIP "${existing_fingerprint}" existing_fingerprint)
    if(existing_fingerprint STREQUAL fingerprint)
      set(rebuild FALSE)
    endif()
  endif()

  if(rebuild)
    message(STATUS "Preparing patched GGML ${source_revision}")
    file(REMOVE_RECURSE "${PREPARE_OUTPUT_DIR}")
    file(MAKE_DIRECTORY "${PREPARE_OUTPUT_DIR}")
    file(COPY "${PREPARE_SOURCE_DIR}/" DESTINATION "${PREPARE_OUTPUT_DIR}"
      PATTERN ".git" EXCLUDE)
    foreach(patch IN LISTS patch_files)
      get_filename_component(patch_name "${patch}" NAME)
      message(STATUS "Applying GGML patch ${patch_name}")
      execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
          "GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}"
          "${GIT_EXECUTABLE}" -C "${PREPARE_OUTPUT_DIR}" apply
          --check --verbose --whitespace=nowarn "${patch}"
        RESULT_VARIABLE check_result
        OUTPUT_VARIABLE check_output
        ERROR_VARIABLE check_error)
      if(NOT check_result EQUAL 0)
        message(FATAL_ERROR
          "GGML patch check failed for ${patch_name}:\n${check_output}${check_error}")
      endif()
      execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
          "GIT_CEILING_DIRECTORIES=${CMAKE_BINARY_DIR}"
          "${GIT_EXECUTABLE}" -C "${PREPARE_OUTPUT_DIR}" apply
          --verbose --whitespace=nowarn "${patch}"
        RESULT_VARIABLE apply_result
        OUTPUT_VARIABLE apply_output
        ERROR_VARIABLE apply_error)
      if(NOT apply_result EQUAL 0)
        message(FATAL_ERROR
          "GGML patch application failed for ${patch_name}:\n${apply_output}${apply_error}")
      endif()
    endforeach()
    # Prevent GGML's version probe from walking out of the copied tree and
    # reporting the parent skin-tokens.cpp commit as its own revision.
    file(WRITE "${PREPARE_OUTPUT_DIR}/.git"
      "gitdir: ${PREPARE_OUTPUT_DIR}/.skintokens-no-git\n")
    file(WRITE "${stamp}" "${fingerprint}\n")
  else()
    message(STATUS "Reusing patched GGML ${source_revision}")
  endif()

  set(${output_variable} "${PREPARE_OUTPUT_DIR}" PARENT_SCOPE)
endfunction()
