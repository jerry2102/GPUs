#include <future>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <memory>
#include <functional>
#include <iostream>

using namespace std;

class ThreadPool {
public:
    ThreadPool(int thread_cnt);
    ~ThreadPool();


    using TaskType = std::function<void()>;
    template<typename F, typename ...Args>
    std::future<typename std::result_of<F(Args...)>::type> enqueue(F&& f, Args&& ...args);

private:
    int _thread_cnt;
    std::vector<std::thread> workers;
    std::deque<TaskType> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool stopped;

};

ThreadPool::ThreadPool(int thread_cnt) : _thread_cnt(thread_cnt), stopped(false) {

    for (int i = 0; i < _thread_cnt; ++i) {
        workers.emplace_back([this] () {
            while (1) {
                TaskType task;
                {
                    std::unique_lock<std::mutex> lock(mtx);
                    this->cv.wait(lock, [this]() { return this->stopped || !tasks.empty();});
                    if (stopped && tasks.empty()) break;
                    task = std::move(tasks.front());
                    tasks.pop_front();

                }
                task();
            }

        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(mtx);
        stopped = true;
    }
    cv.notify_all();
    for (int i = 0; i < workers.size(); i++) {
        workers[i].join();
    }

}

template<typename F, typename... Args>
std::future<typename std::result_of<F(Args...)>::type> ThreadPool::enqueue(F&& f, Args&& ...args)  {

    using TaskReturnType = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<TaskReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<TaskReturnType> res = task->get_future();

    {
        std::unique_lock<std::mutex> lock(mtx);
        tasks.emplace_back([task]() -> void {(*task)();});
    }
    cv.notify_one();
    return res;
}

int main() {
        ThreadPool tpool(3);

        cout << "main thread id: " << pthread_self() << endl;
        for (int i = 0; i < 1; i++) {
                tpool.enqueue([i](int v) {cout << "taskidx: " << i << "v:" << v << ", pid" << pthread_self() << endl;}, i+1);
        }
}