#include "xhand_control.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "data_type.hpp"

namespace py = pybind11;
using namespace xhand_control;

PYBIND11_MODULE(xhand_control, m) {
  py::class_<XHandControl>(m, "XHandControl")
      .def(py::init<>())
      .def("enumerate_devices", &XHandControl::enumerate_devices,
           "Enumerate available devices")
      .def("open_serial", &XHandControl::open_serial,
           "Open a serial device for communication")
      .def("open_ethercat", &XHandControl::open_ethercat,
           "Open a ethercat device for communication")
      .def("read_device_info", &XHandControl::read_device_info,
           "Read device information")
      .def("send_command", &XHandControl::send_command,
           "Send control commands to the device")
      .def("read_state", &XHandControl::read_state,
           "Read device status and other parameters")
      .def("read_parameters", &XHandControl::read_parameters,
           "Read joint parameters")
      .def("set_parameters", &XHandControl::set_parameters,
           "Set joint parameters")
      .def("upgrade_device", &XHandControl::upgrade_device,
           "Perform device upgrade")
      .def("calibrate_joint", &XHandControl::calibrate_joint,
           "Perform joint calibration")
      .def("calibrate_joint_by_mold", &XHandControl::calibrate_joint_by_mold,
           "Perform joint calibration by mold")
      .def("reset_sensor", &XHandControl::reset_sensor, "reset sensors")
      .def("close_device", &XHandControl::close_device, "close device")
      .def("list_hands_id", &XHandControl::list_hands_id, "list hands id")
      .def("get_hand_type", &XHandControl::get_hand_type, "get_hand_type")
      .def("get_serial_number", &XHandControl::get_serial_number,
           "get_serial_number")
      .def("set_hand_name", &XHandControl::set_hand_name, "set_hand_name")
      .def("get_hand_name", &XHandControl::get_hand_name, "get_hand_name")
      .def("set_hand_id", &XHandControl::set_hand_id, "set_hand_id")
      .def("read_version", &XHandControl::read_version, "read_version")
      .def("get_sdk_version", &XHandControl::get_sdk_version,
           "get_sdk_version");
  // 绑定 HandParam_t 结构体
  py::class_<HandParam_t>(m, "HandParam_t")
      .def(py::init<>())  // 默认构造函数
      .def_readwrite("position_stretching", &HandParam_t::position_stretching)
      .def_readwrite("position_closing", &HandParam_t::position_closing)
      .def_readwrite("position_zero", &HandParam_t::position_zero)
      .def_readwrite("position_init", &HandParam_t::position_init)
      .def_readwrite("angle_stretching", &HandParam_t::angle_stretching)
      .def_readwrite("angle_closing", &HandParam_t::angle_closing);

  // 绑定 CalibrationParams 结构体
  py::class_<CalibrationParams>(m, "CalibrationParams")
      .def(py::init<>())  // 默认构造函数
      .def_readwrite("tools", &CalibrationParams::tools)
      .def_readwrite("step", &CalibrationParams::step)
      .def_readwrite("calibrated_angles", &CalibrationParams::calibrated_angles)
      .def_readwrite("lower_angles", &CalibrationParams::lower_angles)
      .def_readwrite("upper_angles", &CalibrationParams::upper_angles);

  // 绑定 CalibrationUnion 联合体
  py::class_<CalibrationUnion>(m, "CalibrationUnion")
      .def(py::init<>())  // 默认构造函数
      .def_readwrite("handparam", &CalibrationUnion::handparam)
      .def_readwrite("calibr_params", &CalibrationUnion::calibr_params);

  // 绑定 DeviceInfo_t 结构体
  py::class_<DeviceInfo_t>(m, "DeviceInfo_t")
      .def(py::init<>())  // 默认构造函数
      .def_readwrite("resverse", &DeviceInfo_t::resverse)
      .def_readwrite("ev_hand", &DeviceInfo_t::ev_hand)
      .def_readwrite("hand_id", &DeviceInfo_t::hand_id)
      .def_readwrite("serial_number", &DeviceInfo_t::serial_number)
      .def_readwrite("name", &DeviceInfo_t::name)
      .def_readwrite("hand_param", &DeviceInfo_t::hand_param)
      .def_readwrite("iap_flag", &DeviceInfo_t::iap_flag)
      .def_readwrite("is_calibrated", &DeviceInfo_t::is_calibrated);

  // 绑定 FingerState_t 结构体
  py::class_<FingerState_t>(m, "FingerState_t")
      .def(py::init<>())  // 默认构造函数
      .def_readwrite("id", &FingerState_t::id)
      .def_readwrite("sensor_id", &FingerState_t::sensor_id)
      .def_readwrite("position", &FingerState_t::position)
      .def_readwrite("torque", &FingerState_t::torque)
      .def_readwrite("raw_position", &FingerState_t::raw_position)
      .def_readwrite("temperature", &FingerState_t::temperature)
      .def_readwrite("commboard_err", &FingerState_t::commboard_err)
      .def_readwrite("jonitboard_err", &FingerState_t::jonitboard_err)
      .def_readwrite("tipboard_err", &FingerState_t::tipboard_err)
      .def_readwrite("default5", &FingerState_t::default5)
      .def_readwrite("default6", &FingerState_t::default6)
      .def_readwrite("default7", &FingerState_t::default7);

  // 绑定 CalibrationErrorInfo 结构体
  py::class_<CalibrationErrorInfo>(m, "CalibrationErrorInfo")
      .def(py::init<>())  // 默认构造函数
      .def_readwrite("calib_status", &CalibrationErrorInfo::calib_status)
      .def_readwrite("joint_number", &CalibrationErrorInfo::joint_number)
      .def_readwrite("calibrated_angle", &CalibrationErrorInfo::calibrated_angle)
      .def_readwrite("calibrated_adc_value", &CalibrationErrorInfo::calibrated_adc_value)
      .def_readwrite("intercept", &CalibrationErrorInfo::intercept); 

  // 绑定 PXSR_ForceData 结构体
  py::class_<PXSR_ForceData>(m, "PXSR_ForceData")
      .def(py::init<>())  // 默认构造函数
      .def_readwrite("fx", &PXSR_ForceData::fx)
      .def_readwrite("fy", &PXSR_ForceData::fy)
      .def_readwrite("fz", &PXSR_ForceData::fz);

  // 绑定 SenserData_t 结构体
  py::class_<SenserData_t>(m, "SenserData_t")
      .def(py::init<>())  // 默认构造函数
      .def_readwrite("calc_force", &SenserData_t::calc_force)
      .def_readwrite("raw_force", &SenserData_t::raw_force)
      .def_readwrite("temperature", &SenserData_t::temperature)
      .def_readwrite("calc_temperature", &SenserData_t::calc_temperature);

  // 绑定 HandState_t 结构体
  py::class_<HandState_t>(m, "HandState_t")
      .def(py::init<>())  // 默认构造函数
      .def_readwrite("finger_state", &HandState_t::finger_state)
      .def_readwrite("sensor_data", &HandState_t::sensor_data);

  // 绑定 FingerCommand_t 结构体
  py::class_<FingerCommand_t>(m, "FingerCommand_t")
      .def(py::init<>())  // 默认构造函数
      .def_readwrite("id", &FingerCommand_t::id)
      .def_readwrite("kp", &FingerCommand_t::kp)
      .def_readwrite("kd", &FingerCommand_t::kd)
      .def_readwrite("ki", &FingerCommand_t::ki)
      .def_readwrite("position", &FingerCommand_t::position)
      .def_readwrite("tor_max", &FingerCommand_t::tor_max)
      .def_readwrite("mode", &FingerCommand_t::mode)
      .def_readwrite("res0", &FingerCommand_t::res0)
      .def_readwrite("res1", &FingerCommand_t::res1)
      .def_readwrite("res2", &FingerCommand_t::res2)
      .def_readwrite("res3", &FingerCommand_t::res3);
  // 绑定 HandCommand_t 结构体
  py::class_<HandCommand_t>(m, "HandCommand_t")
      .def(py::init<>())  // 默认构造函数
      .def_readwrite("finger_command", &HandCommand_t::finger_command);
  // 绑定 ErrorStruct 结构体
  py::class_<ErrorStruct, ErrorStructPtr>(m, "ErrorStruct")
      .def(py::init<>())  // 默认构造函数
      .def_readwrite("error_code", &ErrorStruct::error_code)
      .def_readwrite("error_message", &ErrorStruct::error_message)
      .def("reset", &ErrorStruct::reset, "reset");
}