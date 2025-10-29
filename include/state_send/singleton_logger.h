#include "spdlog/sinks/basic_file_sink.h" // support for basic file logging
#include "spdlog/spdlog.h"
#include <spdlog/common.h>
#include <string>

// based on meyers singleton
// https://www.modernescpp.com/index.php/creational-patterns-singleton/

class SingletonLogger {
private:
  const std::string log_file;
  const spdlog::level::level_enum log_level;
  std::shared_ptr<spdlog::logger> logger;

  SingletonLogger(const std::string &log_file,
                  spdlog::level::level_enum log_level)
      : log_file(log_file), log_level(log_level) {
    logger = spdlog::basic_logger_mt("logger", log_file);
    logger->set_level(log_level);
  }

  SingletonLogger(const SingletonLogger &) = delete;
  SingletonLogger &operator=(const SingletonLogger &) = delete;

public:
  /* this with actual arguments once, afterwards parameters are ignored */
  static SingletonLogger &
  get_logger(const std::string &log_file = "",
             spdlog::level::level_enum log_level = spdlog::level::off) {
    static SingletonLogger instance(log_file,
                                    log_level); // Thread-safe in C++11+
    return instance;
  }
  inline std::string get_log_file() const { return log_file; }

  inline spdlog::level::level_enum get_log_level() const { return log_level; }

  static inline uint64_t get_timestamp_ns() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
  }

  template <typename... Args>
  void info(fmt::format_string<Args...> fmt, Args &&...args) {
    logger->info(fmt, std::forward<Args>(args)...);
  }
};
