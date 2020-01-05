
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include <iostream>

#include "Nexus.h"
#include "Socket.h"
#include "xstring.h"

using std::cout;
using std::endl;

// In this example, the client will send the server a string, "Joel Leagues"
// Then the server will use a function (lambda) to create a welcome message for the username
// After processing the server will send that welcome message back to the client
// lastly we will have the client print out the servers welcome message after running a copy

int main(int argc, char** argv)
{
    Nexus<>::Start();

    Socket::MTU = 5; // this is very small for demo purposes

    auto listen = []() 
    {
        try {
            Socket socket(Socket::Start::Server);
            socket.verbose = true;

            socket.server.set_function([](const xstring* client_data) -> xstring { return xstring("Welcome: '") + *client_data + "'\n"; });

            socket.server.listen("5555").accept().recv(50).respond().close();
            // accept(4) allows the first 4 chars so "Joel" out of "Joel Leagues"
            // accept()  allows an infinite amount

            // socket.server.listen("5555");
            // socket.server.accept();
            // socket.server.recv(50);
            // socket.server.respond();
            // socket.server.close();
        }
        catch (const std::runtime_error & err) {
            cout << err.what() << endl;
        }
    };

    auto send = []() -> Socket
    {
        Socket socket(Socket::Start::Client);
        socket.verbose = true;

        try {
            socket.client.connect("127.0.0.1", "5555").send("Joel Leagues").recv(50).close();
            // recv() >> will continue to recieve all the bytes
            // recv(6) >> will gather 6 bytes so we will get "welcom" from the message

             //socket.client.connect("127.0.0.1", "5555");
             //socket.client.send("Joel Leagues");
             //socket.client.recv(50);
             //socket.client.close();

        }
        catch (const std::runtime_error& err) {
            cout << err.what() << endl;
        }
        return socket;
    };
    Nexus<>::Add_Job(listen);
    Nexus<>::Sleep(555);

    Socket socket = send();

    cout << "returned data >> " << socket.client.buffer.recv << endl;

    Nexus<>::Stop();
    return 0;
}

