cmake $CMAKE_ARGS \
	-DCMAKE_BUILD_TYPE=Release \
	-DLSL_BUNDLED_PUGIXML=OFF \
	-DLSL_UNIXFOLDERS=ON \
	-DCMAKE_CXX_STANDARD=11 \
	-S . -B build
cmake --build build --config Release -j --target install
