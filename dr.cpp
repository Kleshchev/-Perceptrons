#include <iostream>
#include <math.h>
#include <random>
#include <time.h>
#include <Windows.h>
#include <fstream>
#include <vector>
#include <string>

using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;

vector<string> split(const string &str, const string &delim) // сплит строки в вектор
{
    vector<string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == string::npos)
            pos = str.length();
        string token = str.substr(prev, pos - prev);
        if (!token.empty())
            tokens.push_back(token);
        prev = pos + delim.length();
    } while (pos < str.length() && prev < str.length());
    return tokens;
}
void ClearScreen() //очистка экрана
{
    HANDLE hStdOut;
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    DWORD count;
    DWORD cellCount;
    COORD homeCoords = {0, 0};

    hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hStdOut == INVALID_HANDLE_VALUE)
        return;

    /* Get the number of cells in the current buffer */
    if (!GetConsoleScreenBufferInfo(hStdOut, &csbi))
        return;
    cellCount = csbi.dwSize.X * csbi.dwSize.Y;

    /* Fill the entire buffer with spaces */
    if (!FillConsoleOutputCharacter(
            hStdOut,
            (TCHAR)' ',
            cellCount,
            homeCoords,
            &count))
        return;

    /* Fill the entire buffer with the current colors and attributes */
    if (!FillConsoleOutputAttribute(
            hStdOut,
            csbi.wAttributes,
            cellCount,
            homeCoords,
            &count))
        return;

    /* Move the cursor home */
    SetConsoleCursorPosition(hStdOut, homeCoords);
}

int data[4][2] = {{0, 0},
                  {0, 1},
                  {1, 0},
                  {1, 1}}; // входные данные
int answers[4] = {1,
                  0,
                  0,
                  1};    //то, что нам необходимо получить
float weights1[3];       //веса первого входа
float weights2[3];       //веса второго входа
float outputweights[3];  //веса внутреннего слоя
int itcount = 100;       //количество итераций до выхода по ошибке
const float eta = 0.05;  //скорость обучения
const int MAX_RAND = 5;  //максимальное значение случайного числа
const int MIN_RAND = -5; //минимальное значение случайного числа
const int speed = 20;    //скорость отображения в косноли(время одной итерации)
int Activation(float in) //функция для активации нейрона
{
    if (in < 0)
        return 0;
    else
        return 1;
}
int main()
{
    srand(time(0)); //сид рандома из времени

    ifstream iff("weights.txt");
    if (iff.is_open()) //чтение весов из файла, если он есть
    {
        string line;
        int i = 0;
        while (getline(iff, line))
        {
            vector<string> ss = split(line, " ");
            weights1[i] = stof(ss[0]);
            weights2[i] = stof(ss[1]);
            outputweights[i] = stof(ss[2]);
            i++;
            if (i > 2)
                break;
        }
        iff.close();
    }
    else //иначе нахождение правильных весов
    {
        int epocha = 0; //количество итераций
        while (true)
        {
            for (int i = 0; i < 3; i++) //задание рандомных весов
            {
                weights1[i] = rand() % (MAX_RAND - MIN_RAND + 1) + MIN_RAND;
                weights2[i] = rand() % (MAX_RAND - MIN_RAND + 1) + MIN_RAND;
                outputweights[i] = rand() % (MAX_RAND - MIN_RAND + 1) + MIN_RAND;
            }

            int epochaerror = 0; //ошибка данного набора весов
            int hidden[3];       //активация скрытого слоя нейронов
            for (int i = 0; i < itcount; i++)
            {
                ClearScreen();
                epochaerror = 0;
                for (int j = 0; j < 4; j++)
                {
                    float answer = 0;
                    for (int k = 0; k < 3; k++)
                    {
                        hidden[k] = Activation(data[j][0] * weights1[k] + data[j][1] * weights2[k]); //активация скрытого слоя
                    }
                    for (int k = 0; k < 3; k++)
                    {
                        answer += outputweights[k] * hidden[k];
                    }

                    answer = Activation(answer);      //вычисление значения функции
                    float diff = answers[j] - answer; //разница между вычисленным и правильным
                    epochaerror += abs(diff);         //обновление суммы ошибок
                    if (epochaerror != 0)             //обновление весов по дельта правилу, мне кажется именно здесь ошибка, т.к. веса просто начинают расти по модулю не уменьшая конечную ошибку
                    {
                        for (int k = 0; k < 3; k++)
                        {
                            weights1[k] += eta * diff * data[j][0];     //
                            weights2[k] += eta * diff * data[j][1];     //либо в скрытых, либо во входных нейронах нерправильная зависимость т.к в
                            outputweights[k] += eta * diff * hidden[k]; //конечном итоге на выходе (0,1,1,1) вместо (0,1,1,0) и из-за вхожа (1,1) веса входного слоя просто растут по модулю,
                                                                        //не корректируя выход в правильную сторону
                        }
                    }
                    cout << data[j][0] << " " << data[j][1] << "| " << answer << " > " << answers[j] << endl;
                }
                for (int j = 0; j < 3; j++)
                {
                    cout << weights1[j] << " " << weights2[j] << " " << outputweights[j] << endl;
                }
                cout << epochaerror << " - " << epocha * itcount + i << " it" << endl;
                if (epochaerror == 0)
                    break;
                Sleep(speed);
            }
            if (epochaerror == 0)
                break;
            epocha++;
        }
        ofstream out("weights.txt"); //запись правильных весов в файл
        if (out.is_open())
        {
            for (int j = 0; j < 3; j++)
            {
                out << weights1[j] << " " << weights2[j] << " " << outputweights[j] << endl;
            }
            out.close();
        }
    }
    //вычисление значения функции уже с правильными весами либо из файла, либо вычисленными при обучении
    int hidden[3];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            hidden[j] = Activation(data[i][0] * weights1[j] + data[i][1] * weights2[j]);
        }
        float answer = 0;
        for (int j = 0; j < 3; j++)
        {
            answer += hidden[j] * outputweights[j];
        }
        answer = Activation(answer);
        cout << data[i][0] << " " << data[i][1] << "| " << answer << " > " << answers[i] << endl;
    }

    system("pause");
}
