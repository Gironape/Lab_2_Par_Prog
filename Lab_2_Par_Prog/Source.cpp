#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <random>
#include <stdexcept>
#include <omp.h>

std::vector<std::vector<int>> generateRandomMatrix(int rows, int cols) {
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);

#pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }

    return matrix;
}

std::vector<std::vector<int>> readMatrix(const std::string& filename, int& rows, int& cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл.");
    }

    file >> rows >> cols;
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix[i][j];
        }
    }

    file.close();
    return matrix;
}

void writeMatrix(const std::vector<std::vector<int>>& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл для записи.");
    }

    file << matrix.size() << " " << matrix[0].size() << std::endl;

    for (const auto& row : matrix) {
        for (const int& val : row) {
            file << val << " ";
        }
        file << std::endl;
    }

    file.close();
}

std::vector<std::vector<int>> multiplyMatrices(const std::vector<std::vector<int>>& matrix1, const std::vector<std::vector<int>>& matrix2) {
    int rowsA = matrix1.size();
    int colsA = matrix1[0].size();
    int rowsB = matrix2.size();
    int colsB = matrix2[0].size();

    if (colsA != rowsB) {
        throw std::runtime_error("Incorrect matrix sizes for multiplication.");
    }

    std::vector<std::vector<int>> result(rowsA, std::vector<int>(colsB, 0));

#pragma omp parallel for collapse(2)
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

int main() {
    int rows1 = 10;
    int cols1 = 10;
    int rows2 = 10;
    int cols2 = 10;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<int>> matrix1 = generateRandomMatrix(rows1, cols1);
    std::vector<std::vector<int>> matrix2 = generateRandomMatrix(rows2, cols2);

    writeMatrix(matrix1, "matrix1.txt");
    writeMatrix(matrix2, "matrix2.txt");

    std::vector<std::vector<int>> result = multiplyMatrices(matrix1, matrix2);

    writeMatrix(result, "result_matrix.txt");

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "The scope of the task: " << rows1 * cols1 + rows2 * cols2 << " elements." << std::endl;
    std::cout << "Execution time: " << duration.count() << " seconds." << std::endl;

    return 0;
}
