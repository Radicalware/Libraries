#pragma once

#include "Abstract.h"

#include "Macros.h"

namespace Shape
{
    class Rectangle;
    class Circle;
    class Sprite;
};

namespace AbstractFunction
{
    void Draw(const Shape::Circle& FoCircle);
    void Draw(const Shape::Rectangle& FoRectangle);
    void Draw(const Shape::Sprite& FoSprite);
}

class Drawable
{
private:
    AbsStepOneCreateConcept
        AbsCreateFunctionStep1(Draw);
    AbsStepTwoCreateModule
        AbsCreateFunctionStep2(Draw);
    AbsStepThreeCreateSmartPtr

public:
    // --- ABS Functions ---------------------------------------------------------------
    CreateAbstractConstructors(Drawable);
    CreateAbstractFunction(Draw);
    // --- Non-ABS Functions -----------------------------------------------------------
    void SetName(const char* AbstractFunctionName) { NameStr = AbstractFunctionName; }
    void PrintText() { cout << "Object Name: " << NameStr << '\n'; }
    // ---------------------------------------------------------------------------------
private:
    xstring NameStr;
};

namespace Shape
{
    class Circle
    {
    public:
        Circle(int d) : MnDiameter(d) {}
        DefaultConstruct(Circle);
        AbsConstruct(Drawable, Circle);

        int GetCircleDiameter() const { return MnDiameter; }
        void CircleDiameter() { cout << "diameter = " << MnDiameter << endl; }

        void Copy(const Drawable& Other)
        {
            MnDiameter = Other.Cast<Circle>().GetCircleDiameter();
        }
    private:
        int MnDiameter;
    };

    class Rectangle
    {
    public:
        Rectangle(int FnHeight, int FnWidth) : MnHeight(FnHeight), MnWidth(FnWidth) {}
        DefaultConstruct(Rectangle);
        AbsConstruct(Drawable, Rectangle);

        auto GetHeight() const { return MnHeight; }
        auto GetWidth()  const { return MnWidth;  }

        void Copy(const Drawable& Other)
        {
            auto& LoOther = Other.Cast<Rectangle>();
            MnHeight = LoOther.GetHeight();
            MnWidth = LoOther.GetWidth();
        }
    private:
        int MnHeight = 0;
        int MnWidth = 0;
    };


    class Sprite
    {
    public:
        Sprite(const xstring& FsPath) :MsPath(FsPath) {}
        Sprite(const char* FsPath) :MsPath(FsPath) {}
        DefaultConstruct(Sprite);
        AbsConstruct(Drawable, Sprite);

        auto& GetPath() const { return MsPath; }
        auto& GetPath()       { return MsPath; }
        void Copy(const Drawable& Other)
        {
            MsPath = Other.Cast<Sprite>().GetPath();
        }
    private:
        xstring MsPath;
    };
}

namespace AbstractFunction
{
    void Draw(const Shape::Circle& FoCircle)
    {
        std::cout << "Circle Diameter = " << FoCircle.GetCircleDiameter() << std::endl;
    }
    void Draw(const Shape::Rectangle& FoRectangle) {
        std::cout << "Rectangle Dim = [" << FoRectangle.GetHeight() << "," << FoRectangle.GetWidth() << "]" << std::endl;
    }
    void Draw(const Shape::Sprite& FoSprite)
    {
        std::cout << "Drawing sprite = " << FoSprite.GetPath() << std::endl;
    }
}


namespace Abstract
{
    void Run()
    {
        std::vector<Drawable> Objects;
        Objects.push_back(Shape::Circle(10));
        Objects.push_back(Shape::Circle(10));
        Objects.push_back(Shape::Rectangle(12, 42)); // First Object Added
        Objects.push_back(Shape::Sprite("assets/monster.png"));

        for (const Drawable& obj : Objects)
            obj.Draw(); // Call the polymorphic method

        Drawable& LoCircle = Objects[1];
        LoCircle.Cast<Shape::Circle>().CircleDiameter(); // Short Way

        LoCircle.SetName("Circle"); // Base Class (Non-ABS)

        cout << "--------------------------------------" << endl;
        LoCircle.PrintText(); // Base Class (Non-ABS)
        LoCircle.Draw();      // Base Class (ABS)
        LoCircle.Cast<Shape::Circle>().CircleDiameter(); // Derived Class (Non-ABS)
        cout << "--------------------------------------" << endl;
        Drawable LoNewCircle = LoCircle.Cast<Shape::Circle>();
        cout << "--------------------------------------" << endl;
        LoNewCircle.PrintText(); // Base Class (Non-ABS)
        LoNewCircle.Draw();      // Base Class (ABS)
        LoNewCircle.Cast<Shape::Circle>().CircleDiameter(); // Derived Class (Non-ABS)
        cout << "--------------------------------------" << endl;
        Drawable LoSprite = Shape::Sprite("NewPath/Image.png");
        cout << "--------------------------------------" << endl;
        LoSprite.SetName("Sprite"); // Base Class (Non-ABS)
        LoSprite.PrintText(); // Base Class (Non-ABS)
        LoSprite.Draw();      // Base Class (ABS)
        cout << "--------------------------------------" << endl;
    }
}