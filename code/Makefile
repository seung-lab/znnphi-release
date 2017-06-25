# -*- Makefile -*-

HERE    	=       .
BINDIR		=	./bin

AT	=	@
DOLLAR  = 	$$

CD	=	$(AT)cd
CP	=	$(AT)cp
ECHO	=	@echo
CAT	=	$(AT)cat
IF	=	$(AT)if
LN	=	$(AT)ln
MKDIR	=	$(AT)mkdir
MV	=	$(AT)mv
SED	=	$(AT)sed
RM	=	$(AT)rm
TOUCH	=	$(AT)touch
CHMOD	=	$(AT)chmod
DATE    =       $(AT)date

# compilers

ifneq ($(strip $(TUNE)),)
	ZNN_TUNE	=	-mtune=$(strip $(TUNE))
else
	ZNN_TUNE	=
endif


ifneq ($(strip $(ICC)),)
	SSE_FLAGS	=	-DZNN_SSE -msse $(ZNN_TUNE)
	AVX_FLAGS 	= 	-DZNN_AVX -xAVX $(ZNN_TUNE)
	AVX2_FLAGS 	=	-DZNN_AVX2 -march=core-avx2 $(ZNN_TUNE)
	AVX512_FLAGS	=	-DZNN_AVX512 -xMIC-AVX512 $(ZNN_TUNE)
	KNC_FLAGS	=	-DZNN_KNC -mmic $(ZNN_TUNE)
else
	SSE_FLAGS	=	-DZNN_SSE -msse $(ZNN_TUNE)
	AVX_FLAGS	=	-DZNN_AVX -mavx $(ZNN_TUNE)
	AVX2_FLAGS	=	-DZNN_AVX2 -mavx2 -mfma $(ZNN_TUNE)
	AVX512_FLAGS	=	-DZNN_AVX512 -mavx512f -mavx512pf -mavx512cd -mfma $(ZNN_TUNE)
	KNC_FLAGS	=	-DZNN_AVX512 -mavx512f -mavx512pf -mfma $(ZNN_TUNE)
endif

ifneq ($(strip $(CORES)),)
	CORES_FLAG 	=	-DZNN_NUM_CORES=$(strip $(CORES))
else
	CORES_FLAG	=
endif

ifneq ($(strip $(STD1Y)),)
	STANDARD_FLAG	=	-std=c++1y
	STD_FLAG	=	-DZNN_STD1Y=1
else
	STANDARD_FLAG	=	-std=c++14
	STD_FLAG	=
endif


ifneq ($(strip $(ICC)),)
	CXX 			=	$(AT)icpc $(STANDARD_FLAG)
	OPTIMIZATION_FLAGS	=	-DNDEBUG -O3 -ipo -fp-model fast=2  -mkl=sequential -fno-rtti -fno-exceptions
	CS_FLAGS		=	-mkl=sequential
	CS_LD_FLAGS		=	-mkl=sequential  -static-intel
else
	ifneq ($(strip $(GCC5)),)
		CXX			=	$(AT)g++-5 -pthread $(STANDARD_FLAG)
	else
		CXX			=	$(AT)g++ -pthread $(STANDARD_FLAG)
	endif
	OPTIMIZATION_FLAGS	=	-DNDEBUG -O3 -ffast-math -fno-omit-frame-pointer -fno-rtti -fno-exceptions
	CS_FLAGS		=
	CS_LD_FLAGS		=       -lpthread
endif

FPIC	=	-fPIC

CXXINCLUDES	=	-I$(HERE)/include -I$(HERE)/..
CXXWARN		=	-Wall -Wextra -Wno-format -Wno-unknown-pragmas

ifneq ($(strip $(NOHBW)),)
	HBW_LD_FLAG	=
	HBW_FLAG	=	-DZNN_NO_HBW
else
	HBW_LD_FLAG	=	-lmemkind
	HBW_FLAG	=
endif


DEPFLAGS	=	-MM -MG -MP $(CXXINCLUDES) -MT "$(@:.d=.o)" $(CS_FLAGS) -DZNN_NO_CUDA $(HBW_FLAG) $(CORES_FLAG)
INLINE_DEPFLAGS	=	-MMD -MP -MT "$(@)" -MF $(@:.o=.T) -DZNN_NO_CUDA $(HBW_FLAG) $(CORES_FLAG) $(STD_FLAG)


COMMON_FLAGS	=	-g $(INLINE_DEPFLAGS) $(CXXINCLUDES) $(CXXWARN) $(HBW_FLAG) $(CORES_FLAG)


DBG_FLAGS		=	$(COMMON_FLAGS) $(CS_FLAGS)
LD_OPTIMIZATION_FLAGS	=	-DNDEBUG -O3
#-fno-rtti -ffast-math -fno-omit-frame-pointer -fno-exceptions

OPT_FLAGS		=	$(COMMON_FLAGS) $(OPTIMIZATION_FLAGS) $(CS_FLAGS)

COMMON_LDFLAGS		=	$(HBW_LD_FLAG)

DBG_LDFLAGS		=	$(COMMON_LDFLAGS) $(CS_LD_FLAGS)
OPT_LDFLAGS		=	$(COMMON_LDFLAGS) $(CS_LD_FLAGS) $(LD_OPTIMIZATION_FLAGS)

ifneq ($(strip $(OPT)),)
  CXXFLAGS	=	$(OPT_FLAGS)
  CXXLDFLAGS	=	$(OPT_LDFLAGS)
else
  CXXFLAGS	=	$(DBG_FLAGS)
  CXXLDFLAGS	=	$(DBG_LDFLAGS)
endif


OBJS		=	SRC=$(wildcard *.cpp)
SSE_DEPS	=	$(patsubst %.cpp,$(BINDIR)/sse/obj/%.d,$(OBJS))
AVX_DEPS	=	$(patsubst %.cpp,$(BINDIR)/avx/obj/%.d,$(OBJS))
AVX2_DEPS	=	$(patsubst %.cpp,$(BINDIR)/avx2/obj/%.d,$(OBJS))
AVX512_DEPS	=	$(patsubst %.cpp,$(BINDIR)/avx512/obj/%.d,$(OBJS))
KNC_DEPS	=	$(patsubst %.cpp,$(BINDIR)/knc/obj/%.d,$(OBJS))


#OBJS = test3x.o test2x.o test1x.o test1.o lalu.o lala.o lala2.o bench_update_vgg_conv2.o bench_update_vgg_conv3.o  bench_update_vgg_conv4.o
#DEPS =
# dependency files for c++
$(BINDIR)/sse/obj/%.d: ./%.cpp
	$(ECHO) "[CXX] dependencies $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) $(DEPFLAGS) $(SSE_FLAGS) -MF $@ $<

# dependency files for c++
$(BINDIR)/avx/obj/%.d: ./%.cpp
	$(ECHO) "[CXX] dependencies $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) $(DEPFLAGS) $(AVX_FLAGS) -MF $@ $<

# dependency files for c++
$(BINDIR)/avx2/obj/%.d: ./%.cpp
	$(ECHO) "[CXX] dependencies $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) $(DEPFLAGS) $(AVX2_FLAGS) -MF $@ $<

# dependency files for c++
$(BINDIR)/avx512/obj/%.d: ./%.cpp
	$(ECHO) "[CXX] dependencies $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) $(DEPFLAGS) $(AVX512_FLAGS) -MF $@ $<

# dependency files for c++
$(BINDIR)/knc/obj/%.d: ./%.cpp
	$(ECHO) "[CXX] dependencies $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) $(DEPFLAGS) $(KNC_FLAGS) -MF $@ $<

# c++
$(BINDIR)/sse/obj/%.o: ./%.cpp
	$(ECHO) "[CXX] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -c $(CXXFLAGS) $(SSE_FLAGS) -o $@ $<
	$(MV) -f "$(@:.o=.T)" "$(@:.o=.d)"

# c++
$(BINDIR)/avx/obj/%.o: ./%.cpp
	$(ECHO) "[CXX] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -c $(CXXFLAGS) $(AVX_FLAGS) -o $@ $<
	$(MV) -f "$(@:.o=.T)" "$(@:.o=.d)"


# c++
$(BINDIR)/avx2/obj/%.o: ./%.cpp
	$(ECHO) "[CXX] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -c $(CXXFLAGS) $(AVX2_FLAGS) -o $@ $<
	$(MV) -f "$(@:.o=.T)" "$(@:.o=.d)"

# c++
$(BINDIR)/avx512/obj/%.o: ./%.cpp
	$(ECHO) "[CXX] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(ECHO) $(CXX) -c $(CXXFLAGS) $(AVX512_FLAGS) -o $@ $<
	$(CXX) -c $(CXXFLAGS) $(AVX512_FLAGS) -o $@ $<
	$(MV) -f "$(@:.o=.T)" "$(@:.o=.d)"

# c++
$(BINDIR)/knc/obj/%.o: ./%.cpp
	$(ECHO) "[CXX] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -c $(CXXFLAGS) $(KNC_FLAGS) -o $@ $<
	$(MV) -f "$(@:.o=.T)" "$(@:.o=.d)"



# c++
$(BINDIR)/sse/obj/%.S: ./%.cpp
	$(ECHO) "[CXX] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -S $(CXXFLAGS) $(SSE_FLAGS) -o $@ $<

# c++
$(BINDIR)/avx/obj/%.S: ./%.cpp
	$(ECHO) "[CXX] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -S $(CXXFLAGS) $(AVX_FLAGS) -o $@ $<

# c++
$(BINDIR)/avx2/obj/%.S: ./%.cpp
	$(ECHO) "[CXX] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -S $(CXXFLAGS) $(AVX2_FLAGS) -o $@ $<

# c++
$(BINDIR)/avx512/obj/%.S: ./%.cpp
	$(ECHO) "[CXX] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -S $(CXXFLAGS) $(AVX512_FLAGS) -o $@ $<
# c++
$(BINDIR)/knc/obj/%.S: ./%.cpp
	$(ECHO) "[CXX] compiling $<"
	$(MKDIR) -p $(dir $@)
	$(CXX) -S $(CXXFLAGS) $(KNC_FLAGS) -o $@ $<

$(BINDIR)/sse/%.bin: $(BINDIR)/sse/obj/%.o
	$(ECHO) "[CXX] linking $^"
	$(ECHO) $(CXX) $^ $(CXXLDFLAGS) $(AVX_FLAGS) -o $@
	$(CXX) $^ $(CXXLDFLAGS) $(SSE_FLAGS) -o $@

$(BINDIR)/avx/%.bin: $(BINDIR)/avx/obj/%.o
	$(ECHO) "[CXX] linking $^"
	$(ECHO) $(CXX) $^ $(CXXLDFLAGS) $(AVX_FLAGS) -o $@
	$(CXX) $^ $(CXXLDFLAGS) $(AVX_FLAGS) -o $@

$(BINDIR)/avx2/%.bin: $(BINDIR)/avx2/obj/%.o
	$(ECHO) "[CXX] linking $^"
	$(ECHO) $(CXX) $^ $(CXXLDFLAGS) $(AVX2_FLAGS) -o $@
	$(CXX) $^ $(CXXLDFLAGS) $(AVX2_FLAGS) -o $@

$(BINDIR)/avx512/%.bin: $(BINDIR)/avx512/obj/%.o
	$(ECHO) "[CXX] linking $^"
	$(ECHO) $(CXX) $^ $(CXXLDFLAGS) $(AVX512_FLAGS) -o $@
	$(CXX) $^ $(CXXLDFLAGS) $(AVX512_FLAGS) -o $@

$(BINDIR)/knc/%.bin: $(BINDIR)/knc/obj/%.o
	$(ECHO) "[CXX] linking $^"
	$(ECHO) $(CXX) $^ $(CXXLDFLAGS) $(KNC_FLAGS) -o $@
	$(CXX) $^ $(CXXLDFLAGS) $(KNC_FLAGS) -o $@


$(BINDIR)/avx512/%.report: ./%.cpp
	$(ECHO) "[CXX] optimization report $<"
	$(MKDIR) -p $(dir $@)
	$(ECHO) $(CXX) -qopt-report=4 $(CXXFLAGS) $(AVX512_FLAGS)-o $@ $<
	$(CXX) -qopt-report=4 $(CXXFLAGS) $(CXXLDFLAGS) $(AVX512_FLAGS) -o $@ $<


.PHONY: all clean remake

all:
	$(ECHO) $(DEPS)

clean:
	$(ECHO) Cleaning...
	$(RM) -rf $(BINDIR)


remake: clean all

#-include $(SSE_DEPS)
#-include $(AVX_DEPS)
#-include $(AVX2_DEPS)
-include $(AVX512_DEPS)
# -include $(KNC_DEPS)
