#pragma once

#include <iostream>

#include "Mirror.h"
using std::cout;
using std::endl;
using std::string;

namespace Park
{
    class Zoo2 : public RA::Mirror<std::string, int>
    {
    public:
        void AddAnimal(   const string& FsAnimal, const int FnIndex) { The.AddKVP(FsAnimal, FnIndex); }
        void RemoveAnimal(const string& FsAnimal) { The.RemoveKey(FsAnimal); }
    };
};

namespace Example
{
    int StringInt()
    {
        auto LoZoo = Park::Zoo2();

        LoZoo.AddAnimal("Bird",   0);
        LoZoo.AddAnimal("Lizard", 1);
        LoZoo.AddAnimal("Gator",  2);

        cout << "Bird, Lizard, Gator" << endl; // Don't count on it being in order
        for (xint i = 0; i < LoZoo.Size(); i++)
            cout << "Animal: " << LoZoo[i] << endl;
        cout << '\n';

        LoZoo.RemoveAnimal("Lizard");

        cout << "Bird, Gator" << endl;
        for (auto& [Key, Value] : LoZoo)
            cout << Key << " : " << Value << endl;
        cout << '\n';

        // Bird, Gator
        LoZoo.AddAnimal("Dog", 3);
        LoZoo.AddAnimal("Cat", 4);
        // Bird, Gator, Dog, Cat
        LoZoo.Replace("Dog", "Parrot");
        // Bird, Gator, Parrot, Cat

        for (auto& [Key, Value] : LoZoo)
            cout << Key << " : " << Value << endl;
        cout << '\n';

        LoZoo.Replace("Gator", "Monkey");

        for (auto& [Key, Value] : LoZoo)
            cout << Key << " : " << Value << endl;
        cout << '\n';

        return 0;
    }
}
