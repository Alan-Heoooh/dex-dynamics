#ifndef _ERROR_MANAGER_HPP_
#define _ERROR_MANAGER_HPP_
#include <iostream>
#include <map>
#include <memory>
#include <string>

namespace xhand_control {

struct ErrorStruct {
  ErrorStruct(){};
  ErrorStruct(int code, std::string message) {
    error_code = code;
    error_message = message;
  }
  int error_code{0};
  std::string error_message;
  void reset() {
    error_code = 0;
    error_message = "";
  }
  // 重载 bool 运算符
  explicit operator bool() const { return error_code == 0; }
};
using ErrorStructPtr = std::shared_ptr<ErrorStruct>;
// 定义错误码枚举
enum class ErrorCode : int {
  ERROR_ID = 1501001,
  ERROR_CMD = 1501002,
  ERROR_COMMUNICATION = 1501003,
  ERROR_DATA_LEN = 1501004,
  ERROR_NOFLASHPARAM = 1501005,
  ERROR_COMMUNICATION_BUSY = 1501006,

  IAPERROR_SIZE = 1501007,
  IAPERROR_ERASE = 1501008,
  IAPERROR_WRITEFLASH = 1501009,
  IAPERROR_INDEX = 1501010,
  IAPERROR_BUSY = 1501011,
  IAPERROR_FORM = 1501012,
  IAPERROR_VERIFY = 1501013,
  // IAPERROR_OTHER = 1501014,
  IAPERROR_SHAR = 1501015,
  IAPERROR_JUMP_CHECK = 1501016,
  IAPERROR_SHA_CHECK = 1501017,

  ERROR_READ_TOTAL_FORCE = 1501018,
  ERROR_READ_FORCES = 1501019,
  ERROR_READ_TEMP = 1501020,
  ERROR_CALIBRATE = 1501021,
  SENSOR_ERROR_CMD = 1501022,
  SENSOR_ERROR_NOFLASHPARAM = 1501023,
  SENSOR_ERROR_COMMUNICATION = 1501024,

  JOINT_ERROR_PARAM_INIT = 1501026,
  JOINT_ERROR_TEMP_PROCTED = 1501028,
  JOINT_ERROR_MOTION_MODE = 1501029,
  JOINT_ERROR_PARAM_OUTOF_RANGE = 1501030,
  JOINT_ERROR_NOFLASHPARAM = 1501031,
  JOINT_ERROR_COMMUNICATION = 1501032,
  JOINT_ERROR_CMD = 1501033,
  JOINT_ERROR_POSITION_RAW = 1501034,
  JOINT_ERROR_CURRENT_PROCTED = 1501035,

  ERROR_BOOT_CMD = 1501036,

  INVALID_DEVICE_ID = 1501037,
  DEVICE_NOT_INIT = 1501038,
  OPEN_DEVICE_FAILED = 1501039,
  PDO_COMMUNICATION_ERROR = 1501040,
  SDO_COMMUNICATION_ERROR = 1501041,
  UPGRADE_STAGE_BEGIN_ERROR = 1501042,
  UPGRADE_STAGE_TRANSMIT_ERROR = 1501043,
  UPGRADE_STAGE_VERIFY_ERROR = 1501044,
  INVALID_FIRMWARE_FILE = 1501045,

  CALIBRATE_OUT_OF_RANGE = 1501046,              // 数值超出范围
  CALIBRATE_NEAR_BOUNDARY = 1501047,             // 数值靠近边界
  CALIBRATE_SMALL_DIFFERENCE = 1501048,          // 差值过小
  CALIBRATE_SMALL_DIFFERENCE_SPECIAL = 1501049,  // 通道 3 差值过小

  SERIAL_COMMUNICATION_ERROR = 1501050,

  FIRMWARE_VERSION_TOO_LOWER = 1501051,               // 固件版本太低     
  CALIBRATION_ERROR_ANGLE_OUT_OF_RANGE = 1501052,     // 角度超出范围
  CALIBRATION_ERROR_ADC_OUT_OF_RANGE = 1501053,       // ADC 超出范围
  CALIBRATION_ERROR_ADC_PRE_UPPER_ERROR = 1501054,    // ADC 预测上边界参数错误
  CALIBRATION_ERROR_ADC_PRE_LOWER_ERROR = 1501055,    // ADC 预测下边界参数错误
  CALIBRATION_SAVE_PARAM_ERROR = 1501056,             // 校准时保存参数错误
  CALIBRATION_INPUT_PARAM_ERROR = 1501057,            // 校准输入参数错误
  CALIBRATION_INVALID_TOOL = 1501058,                 // 传入校准工装型号错误，判断逻辑：S/D开头的pass
};

// 错误管理类
class ErrorManager {
 public:
  // 接收硬件错误码，转换为自定义错误码并返回错误描述
  static ErrorStruct get_error_info(int hardware_error_code);

 private:
  // 硬件错误码到自定义错误码的映射
  static std::map<int, int> hardware_to_custom_error_code_;

  // 自定义错误码到描述的映射
  static std::map<int, std::string> custom_error_message_;
};
}  // namespace xhand_control

#endif  // _ERROR_MANAGER_HPP_