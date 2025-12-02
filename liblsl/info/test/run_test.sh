

set -ex



lslver
test -f $PREFIX/include/lsl_c.h
test -f $PREFIX/include/lsl_cpp.h
test -f $PREFIX/lib/liblsl${SHLIB_EXT}
exit 0
