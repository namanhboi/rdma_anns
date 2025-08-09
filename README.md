To build, 
`cmake -S. -B build -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DTEST_UDL1=ON -DTEST_UDL2=OFF`
`cmake --build build -j`
- TEST\_UDL1 or TEST\_UDL2 to toggle which udl you want to test
