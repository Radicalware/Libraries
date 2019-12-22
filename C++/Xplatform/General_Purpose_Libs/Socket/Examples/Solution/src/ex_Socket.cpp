#include <iostream>

#include "Nexus.h"
#include "Socket.h"
#include "xstring.h"

using std::cout;
using std::endl;

// In this example, the client will send the server a name "King Conan"
// Then the server will use a function (lambda) to create a welcom message for the name
// then the server will send that welcom message back to the client
// lastly we will have the client print out the servers welcome message

int main(int argc, char** argv)
{
    Nexus<>::Start();

    Socket::MTU = 5; // this is very small for demo purposes

    auto listen = []() 
    {
        try {
            Socket socket(Socket::Start::Server);

            socket.server.set_function([](const xstring* client_data) -> xstring { return xstring("Welcome: '") + *client_data + "'\n"; });

            socket.server.listen("5555").accept().recv(50).respond().close();
            // accept(4) allows the first 4 chars so "King" out of "King Conan"
            // accept()  allows an infinite amount
        }
        catch (const std::runtime_error & err) {
            cout << err.what() << endl;
        }
    };

    auto send = []() -> Socket
    {
        Socket socket(Socket::Start::Client);
        try {
            socket.client.connect("127.0.0.1", "5555").send("King Conan").recv(50).close();
            // recv() >> will continue to recieve all the bytes
            // recv(6) >> will gather 6 bytes so we will get "welcom" from the message
        }
        catch (const std::runtime_error& err) {
            cout << err.what() << endl;
        }
        return socket;
    };
    Nexus<>::Add_Job(listen);
    Nexus<>::Sleep(55);

    Socket socket = send();

    cout << "returned data >> " << socket.client.buffer.recv << endl;

    Nexus<>::Stop();
    return 0;
}

