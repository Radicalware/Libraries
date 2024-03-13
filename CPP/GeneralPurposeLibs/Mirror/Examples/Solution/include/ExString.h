#pragma once

#include <iostream>

#include "Mirror.h"
using std::cout;
using std::endl;
using std::string;

namespace Park
{
    class Zoo : public RA::Mirror<std::string>
    {
    public:
        void AddAnimal(const string& FsAnimal) { return The.AddElement(FsAnimal); }
        void RemoveAnimal(const string& FsAnimal) { return The.RemoveElement(FsAnimal); }
    };
};

namespace Example
{
    int String()
    {
        auto LoZoo = Park::Zoo();

        LoZoo.AddAnimal("Bird");
        LoZoo.AddAnimal("Lizard");
        LoZoo.AddAnimal("Gator");

        cout << "Bird, Lizard, Gator" << endl; // Don't count on it being in order
        for (xint i = 0; i < LoZoo.Size(); i++)
            cout << "Animal: " << LoZoo[i] << endl;
        cout << '\n';

        LoZoo.RemoveAnimal("Lizard");

        cout << "Bird, Gator" << endl;
        for (xint i = 0; i < LoZoo.Size(); i++)
            cout << "Animal: " << LoZoo[i] << endl;
        cout << '\n';

        // Bird, Gator
        LoZoo.AddAnimal("Dog"); // 2
        LoZoo.AddAnimal("Cat"); // 3
        // Bird, Gator, Dog, Cat
        LoZoo.Replace("Dog", "Parrot"); // 2
        // Bird, Gator, Parrot, Cat

        cout << "Bird, Gator, Parrot, Cat" << endl;
        for (xint i = 0; i < LoZoo.Size(); i++)
            cout << "Animal: " << LoZoo[i] << endl;
        cout << '\n';

        LoZoo.Replace(1, "Monkey");

        cout << "Bird, Monkey, Parrot, Cat" << endl;
        for (xint i = 0; i < LoZoo.Size(); i++)
            cout << "Animal: " << LoZoo[i] << " : " << LoZoo[LoZoo[i]] << endl;
        cout << '\n';

        return 0;
    }
}
