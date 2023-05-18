#include <cpprest/http_listener.h>
#include <cpprest/json.h>
#include <thread>
#include <Python.h>
#include <typeinfo>

using namespace web;
using std::cout;
using std::cin;
using namespace http;
using namespace utility;
using namespace http::experimental::listener;

PyObject* get_trans_func(){
    PyObject* pFunc   = NULL;
    PyObject* pModule = NULL;
    
    Py_Initialize();

    PyRun_SimpleString( "import sys" );
    PyRun_SimpleString( "import os" );
    PyRun_SimpleString( "sys.path.append('..')" );

    if (!(pModule = PyImport_ImportModule("model"))){
        std::cout << "Error: Py: Failed to import Python module\n";
        Py_XDECREF(pModule);
        return NULL;
    }
    if (!(pFunc = PyObject_GetAttrString(pModule, "translate"))){
        std::cout << "Error: Py: Failed to import Python function\n";
        Py_XDECREF(pModule);
        return NULL;
    }
    Py_XDECREF(pModule);
    return pFunc;
}

void stop_py(PyObject* trans_func){
    Py_XDECREF(trans_func);
    Py_Finalize();
}



void start_listener_give_translation(PyObject* trans_func) {
    // Создание http_listener на порту 8080 и адресе /get_plus/
    http_listener listener("http://localhost:8080/give_translation/");
    
    // Обработчик GET запросов
    listener.support(methods::GET, [trans_func](http_request request){
        // Получение значения числа из параметра запроса
        auto query = uri::split_query(request.request_uri().query());

        // обработка запроса //
        cout << query["text"] << " <-- запрос\n";
        // Вызов функции Python
        cout << trans_func << "\n";
        // cout << strdup(PyBytes_AS_STRING(PyUnicode_AsEncodedString(PyObject_CallFunction(trans_func, "sss", "text", "laung_from", "laung_to"), "utf-8", "ERROR")));
        PyObject* trans_text_py = PyObject_CallFunction(trans_func, "sss", query["text"].c_str(), query["laung_from"].c_str(), query["laung_to"].c_str());
        cout << trans_text_py << "\n";
        // Преобразование результата в C++ тип
        std::string trans_text_cpp = strdup(PyBytes_AS_STRING(PyUnicode_AsEncodedString(trans_text_py, "utf-8", "ERROR"))); //
        //std::string translate_text = , , ;        

        // Создание JSON объекта с результатом
        json::value result;

        result["translate_text"] = json::value::string(trans_text_cpp);
        result["status"] = 200;
        
        Py_XDECREF(trans_text_py);
        // Отправка ответа в формате JSON
        request.reply(status_codes::OK, result);
    });

    // Запуск http_listener
    listener.open().wait();

    // Бесконечный цикл ожидания запросов
    while (true);
}

int main(int argc, char *argv[]) {

    PyObject* trans_func = get_trans_func();
    // Запуск сервера в отдельном потоке
    std::thread server_thread(start_listener_give_translation, std::ref(trans_func));

    // Ожидание нажатия клавиши Enter для завершения работы
    std::cin.get();

    // Остановка сервера и интерпритатора питона
    server_thread.detach();
    stop_py(trans_func);

    return 0;
}