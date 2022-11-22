
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include <iostream>

#include "Nexus.h"
#include "Socket.h"
#include "xstring.h"
#include "xmap.h"
#include "Macros.h"

using std::cout;
using std::endl;


void DummyExample()
{
    Begin();

    xvector<xstring> LvNums = { "one", "two", "three" };
    xvector<xstring*> LvPtrNums = LvNums.GetPtrs();
    LvPtrNums.ForEachThread([](auto& Val) { return Val + '\n'; }).Join(">> ").Split('\n').Join().Findall(R"(\w+)").Join('\n').Print();
    auto& Val = LvPtrNums[1];

    Rescue();
}




// In this example, the client will send the server a string, "Joel Leagues"
// Then the server will use a function (lambda) to create a welcome message for the username
// After processing the server will send that welcome message back to the client
// lastly we will have the client print out the servers welcome message after running a copy

int main(int argc, char** argv)
{
    Begin();
    Nexus<>::Start();

    Socket::MTU = 5; // this is very small for demo purposes

    auto listen = []() 
    {
        try {
            Socket socket(Socket::Start::Server);
            socket.verbose = true;

            socket.server.SetFunction([](const xstring* client_data) -> xstring { return xstring("Welcome: '") + *client_data + "'\n"; });

            ///socket.server.listen("5555").accept().recv(50).respond().Close();
            // accept(4) allows the first 4 chars so "Joel" out of "Joel Leagues"
            // accept()  allows an infinite amount

             socket.server.Listen("5555");
             socket.server.Accept();
             socket.server.Recv(50);
             socket.server.Respond();
             socket.server.Close();
        }
        catch (const std::runtime_error & err) {
            cout << err.what() << endl;
        }
    };

    auto send = []() -> Socket
    {
        Socket socket(Socket::Start::Client, Socket::Protocol::UDP);
        socket.verbose = true;

        try {
            ///socket.client.Connect("127.0.0.1", "5555").send("Joel Leagues").recv(50).Close();
            // recv() >> will continue to recieve all the bytes
            // recv(6) >> will gather 6 bytes so we will get "welcom" from the message

             socket.client.Connect("127.0.0.1", "5555");
             socket.client.Send("Joel Leagues");
             socket.client.Recv(50);
             socket.client.Close();

        }
        catch (const std::runtime_error& err) {
            cout << err.what() << endl;
        }
        return socket;
    };
    Nexus<>::AddJob(listen);
    Nexus<>::Sleep(555);

    Socket socket = send();

    cout << "returned data >> " << socket.client.buffer.recv << endl;

    RescuePrint();
    Nexus<>::Stop();
    return 0;
}

