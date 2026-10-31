# Runs brusselator_graph_tool with the committed
# reference layout's own arguments and requires its output to be BYTE FOR
# BYTE identical to the committed reference YAML, so the committed text
# never drifts from the builder. Same cmake -P
# driver convention as compare-runs.cmake (this directory).
#
# Invoked by CTest as:
#   cmake -DBIN=... -DOUT=<scratch.yaml> -DREFERENCE=<reference.yaml> \
#         -DARGS="--ranks-u;1;--ranks-v;1;--ranks-coupler;1;--launch;mpmd" \
#         -P this
if(NOT BIN OR NOT OUT OR NOT REFERENCE)
  message(FATAL_ERROR "verify-graph-tool-reference.cmake: BIN, OUT and REFERENCE are required")
endif()

execute_process(COMMAND ${BIN} ${ARGS} --out ${OUT}
                 OUTPUT_VARIABLE _out ERROR_VARIABLE _err RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "verify-graph-tool-reference.cmake: ${BIN} exited ${_rc}\n${_out}\n${_err}")
endif()

execute_process(COMMAND ${CMAKE_COMMAND} -E compare_files ${OUT} ${REFERENCE}
                 RESULT_VARIABLE _diff_rc)
if(NOT _diff_rc EQUAL 0)
  execute_process(COMMAND diff -u ${REFERENCE} ${OUT} OUTPUT_VARIABLE _d)
  message(FATAL_ERROR
    "verify-graph-tool-reference.cmake: ${OUT} (freshly generated) differs "
    "from the committed reference ${REFERENCE} -- the committed YAML has "
    "drifted from what add_window_coupling() now builds:\n${_d}")
endif()

message(STATUS "brusselator_graph_tool reproduces ${REFERENCE} byte for byte")
