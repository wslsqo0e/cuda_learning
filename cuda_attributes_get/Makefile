# c++ compile
INCLUDE_DIR = -isystem /usr/local/cuda-12.8/include/

CUDA_ARCH_FLAGS = -gencode=arch=compute_75,code=sm_75 \
                  -gencode=arch=compute_80,code=sm_80 \
                  -gencode=arch=compute_89,code=sm_89 \
                  -gencode=arch=compute_90a,code=sm_90a

CXX = g++
CXX_FLAGS = -std=c++17 -Wfatal-errors -Wall -Wextra -Wpedantic -Wconversion -Wshadow -march=native \
            $(INCLUDE_DIR)

NVCC = nvcc
NVCC_FLAGS = -std=c++17 --expt-relaxed-constexpr \
             -O3 -use_fast_math --threads=4 \
             ${CUDA_ARCH_FLAGS} \
             $(INCLUDE_DIR)

LINK_FLAGS = ${CUDA_ARCH_FLAGS}

# Final binary
BIN = main
# Put all auto generated stuff to this build dir.
BUILD_DIR = ./build
TEST_BIN_DIR = $(BUILD_DIR)/tests

# List of all .cpp and .cu source files. (excluding test files)
MAIN_CPP_CU_SRCS = $(filter-out %_test.cpp %_test.cu, $(wildcard *.cpp *.cu))

# Test sources are those ending with _test.cpp or _test.cu
TEST_CPP_CU_SRCS = $(filter %_test.cpp %_test.cu, $(wildcard *.cpp *.cu))

# All .o files go to build dir.
# Replace .cpp with .o and .cu with .o, prefixed by BUILD_DIR.
OBJ = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(patsubst %.cu,$(BUILD_DIR)/%.o,$(MAIN_CPP_CU_SRCS)))
TEST_BINS = $(patsubst %_test.cpp,$(TEST_BIN_DIR)/%,$(patsubst %_test.cu,$(TEST_BIN_DIR)/%,$(TEST_CPP_CU_SRCS)))

# Gcc/Clang will create these .d files containing dependencies.
DEP = $(OBJ:%.o=%.d)

# Default target named after the binary.
ALL : $(BUILD_DIR)/$(BIN)

# Actual target of the binary - depends on all .o files.
$(BUILD_DIR)/$(BIN) : $(OBJ)
	# Create build directories - same structure as sources.
	mkdir -p $(@D)
	# Just link all the object files.
	$(NVCC) ${LINK_FLAGS} $^ -o $@

# Include all .d files
-include $(DEP)

# Build target for every single object file.
# The potential dependency on header files is covered
# by calling `-include $(DEP)`.
$(BUILD_DIR)/%.o : %.cpp
	mkdir -p $(@D)
	# The -MMD flags additionaly creates a .d file with
	# the same name as the .o file.
	$(CXX) $(CXX_FLAGS) -MMD -c $< -o $@

$(BUILD_DIR)/%.o : %.cu
	mkdir -p $(@D)
	# --generate-dependencies-with-compile 行为同 -MMD 类似
	$(NVCC) $(NVCC_FLAGS) --generate-dependencies-with-compile -c $< -o $@

.PHONY : test

# 'test' target depends on all individual test executables
test : $(TEST_BINS)
	@echo "--- Running all tests ---"
	@for TEST_BIN in $(TEST_BINS); do \
		echo "Running $$TEST_BIN..."; \
		$$TEST_BIN || { echo "Test $$TEST_BIN FAILED!"; exit 1; }; \
	done
	@echo "--- All tests PASSED ---"


# Rule for building individual test executables from their source files.
# Each *_test.cpp or *_test.cu directly compiles and links to its own executable.
# This assumes each test file is self-contained (i.e., has its own main function).
$(TEST_BIN_DIR)/% : %_test.cpp
	mkdir -p $(@D)
	$(CXX) $(CXX_FLAGS) $< -o $@

$(TEST_BIN_DIR)/% : %_test.cu
	mkdir -p $(@D)
	$(NVCC) ${LINK_FLAGS} $< -o $@

# 允许单独编译执行一个测试用例
% : $(TEST_BIN_DIR)/%
	@echo "--- Running $@ ---"
	$<
	@echo ""

# This should remove all generated files.
.PHONY : clean
clean :
	-rm -rf $(BUILD_DIR)/


.PHONY : run
run : $(BUILD_DIR)/$(BIN)
	$(BUILD_DIR)/$(BIN)
