#include<iostream>

#include "cc.h"

using std::cout;
using std::endl;

int main(){

    cout << "Test Colors\n";
    cout << cc::black << "Test black\n";
    cout << cc::red << "Test red\n";
    cout << cc::green << "Test green\n";
    cout << cc::yellow << "Test yellow\n";
    cout << cc::blue << "Test blue\n";
    cout << cc::magenta << "Test magenta\n";
    cout << cc::cyan << "Test cyan\n";
    cout << cc::grey << "Test grey\n";
    cout << cc::white << "Test white";

    cout << cc::black << "Test black";
    cout << cc::on_black << "\nTest on_black";
    cout << cc::on_red << "\nTest on_red";
    cout << cc::on_green << "\nTest on_green";
    cout << cc::on_yellow << "\nTest on_yellow";
    cout << cc::on_blue << "\nTest on_blue";
    cout << cc::on_magenta << "\nTest on_magenta";
    cout << cc::on_cyan << "\nTest on_cyan";

    cout << cc::on_grey << "\nTest on_grey";
    cout << cc::on_clear << "\nTest on_clear\n";


    cout << cc::on_grey << cc::red << "white background & red text" << cc::reset << '\n' ;

    return 0;
}
