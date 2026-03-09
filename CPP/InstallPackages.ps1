# use --recurse to rebuild a package

set64

vcpkg install --triplet x64-windows `
    boost `
    zlib `
    libbson `
    openssl `
    boost-system `
    boost-date-time `
    boost-regex `
    boost-interprocess `
    websocketpp `
    atlmfc `
    vtk[atlmfc] `
    mongo-cxx-driver `
    vulkan `
    nlohmann-json `
    cpprestsdk `
    glm `
    abseil
