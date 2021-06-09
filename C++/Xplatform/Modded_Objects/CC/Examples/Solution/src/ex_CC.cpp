
// Copyright[2019][Joel Leagues aka Scourge] under the Apache V2 Licence

#include<iostream>

#include "CC.h"

using std::cout;
using std::endl;

int main(){

    cout << "Test Colors\n";
    cout << CC::Black << "Test black\n";
    cout << CC::Red << "Test red\n";
    cout << CC::Green << "Test green\n";
    cout << CC::Yellow << "Test yellow\n";
    cout << CC::Blue << "Test blue\n";
    cout << CC::Magenta << "Test magenta\n";
    cout << CC::Cyan << "Test cyan\n";
    cout << CC::Grey << "Test grey\n";
    cout << CC::White << "Test white";

    cout << CC::Black << "Test black";
    cout << CC::OnBlack << "\nTest Onblack";
    cout << CC::OnRed << "\nTest Onred";
    cout << CC::OnGreen << "\nTest Ongreen";
    cout << CC::OnYellow << "\nTest Onyellow";
    cout << CC::OnBlue << "\nTest Onblue";
    cout << CC::OnMagenta << "\nTest ToOnMagenta";
    cout << CC::OnCyan << "\nTest Oncyan";

    cout << CC::OnGrey << "\nTest Ongrey";
    cout << CC::RemoveBackgroundColor << "\nTest Onclear\n";


    cout << CC::OnGrey << CC::Red << "white background & red text" << CC::Reset << '\n' ;

    return 0;
}
