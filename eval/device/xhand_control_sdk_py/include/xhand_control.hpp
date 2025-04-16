#ifndef _XHAND_CONTROL_HPP__
#define _XHAND_CONTROL_HPP__

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "communiation_interface.hpp"
#include "visibility_control.h"

using namespace communication_interface;
namespace xhand_control {
class XHandControl {
 public:
  XHandControl();
  ~XHandControl() = default;

  /**
   * @brief  Enumerate devices connected to the computer.
   *         This function supports different communication types such as serial
   * and EtherCAT.
   *
   * @param comm_type  The communication type, either "serial" or "ethercat".
   * @return std::vector<std::string>  A list of device names.
   */
  XHAND_CONTROL_PUBLIC std::vector<std::string> enumerate_devices(
      const std::string& comm_type);

  /**
   * @brief  Open a serial communication channel with the device.
   *
   * @param port  The name of the serial port to open.
   * @param baudrate  The baud rate for the serial connection.
   * @return ErrorStruct  Structure indicating success or failure and related
   * error information.
   */
  XHAND_CONTROL_PUBLIC ErrorStruct open_serial(const std::string& port,
                                               uint32_t baudrate);

  /**
   * @brief  Open an EtherCAT communication channel with the device.
   *
   * @param ifname  The EtherCAT network interface name.
   * @return ErrorStruct  Structure indicating success or failure and related
   * error information.
   */
  XHAND_CONTROL_PUBLIC ErrorStruct open_ethercat(const std::string& ifname);

  /**
   * @brief  Get the device information for the specified device ID.
   *
   * @param device_id  The ID of the device to retrieve information from.
   * @return std::pair<ErrorStruct, DeviceInfo_t>  A pair containing error
   * information and device details.
   */
  XHAND_CONTROL_PUBLIC std::pair<ErrorStruct, DeviceInfo_t> read_device_info(
      uint8_t device_id);

  /**
   * @brief  Send a command to the specified device.
   *
   * @param device_id  The ID of the device to send the command to.
   * @param command  The command to be sent to the device.
   * @return ErrorStruct  Structure indicating success or failure and related
   * error information.
   */
  XHAND_CONTROL_PUBLIC ErrorStruct send_command(const uint8_t device_id,
                                                const HandCommand_t& command);

  /**
   * @brief  Read the current state of the specified device.
   *
   * @param device_id  The ID of the device whose state is to be read.
   * @param force_update  If true, force an update of the device state,it need
   * some time.
   * @return std::pair<ErrorStruct, HandState_t>  A pair containing error
   * information and device state.
   */
  XHAND_CONTROL_PUBLIC std::pair<ErrorStruct, HandState_t> read_state(
      uint8_t device_id, bool force_update = false);

  /**
   * @brief  Read the configuration parameters of the specified device.
   *
   * @param device_id  The ID of the device whose parameters are to be read.
   * @return std::pair<ErrorStruct, HandParam_t>  A pair containing error
   * information and device parameters.
   */
  XHAND_CONTROL_PUBLIC std::pair<ErrorStruct, HandParam_t> read_parameters(
      uint8_t device_id);

  /**
   * @brief  Set the configuration parameters for the specified device.
   *
   * @param device_id  The ID of the device whose parameters are to be set.
   * @param parameters  The parameters to be set on the device.
   * @return ErrorStruct  Structure indicating success or failure and related
   * error information.
   */
  XHAND_CONTROL_PUBLIC ErrorStruct
  set_parameters(const uint8_t device_id, const HandParam_t& parameters);

  /**
   * @brief  Upgrade the firmware of the device.
   *
   * @param device_id  The ID of the device to upgrade.
   * @param component_id  The component of the device to upgrade.
   * @param firmware_path  The file path to the firmware to be loaded.
   * @return ErrorStruct  Structure indicating success or failure and related
   * error information.
   */
  XHAND_CONTROL_PUBLIC ErrorStruct
  upgrade_device(const uint8_t device_id, const uint8_t component_id,
                 const std::string& firmware_path);

  /**
   * @brief  Read the version of the firmware or component on the device.
   *
   * @param device_id  The ID of the device.
   * @param component_id  The component whose version is to be read.
   * @return std::pair<ErrorStruct, std::string>  A pair containing error
   * information and the version string.
   */
  XHAND_CONTROL_PUBLIC std::pair<ErrorStruct, std::string> read_version(
      const uint8_t device_id, const uint8_t component_id);

  /**
   * @brief  Calibrate the joints of the device.
   *
   * @param device_id  The ID of the device to calibrate.
   * @param step  The calibration step to perform.
   * @return ErrorStruct  Structure indicating success or failure and related
   * error information.
   */
  XHAND_CONTROL_PUBLIC ErrorStruct
  calibrate_joint(const uint8_t device_id, const uint8_t step,
                  const std::vector<int8_t> angles_limit = {});

  /**
   * @brief  Calibrate the joints of the device with target angles.(override)
   *
   * @param device_id  The ID of the device to calibrate.
   * @param tools  The type of calibration tools.
   * @param step  The calibration step to perform.
   * @param calibrated_angles  The fingures calibration target angles.
   * @param angles_limit  The fingures calibration target angles limit.
   * @return ErrorStruct  Structure indicating success or failure and related
   * error information.
   */
  XHAND_CONTROL_PUBLIC ErrorStruct
  calibrate_joint_by_mold(const uint8_t device_id, const std::string&  tools, const uint8_t step,
                  const std::vector<int8_t> calibrated_angles = {}, 
                  const std::vector<int8_t> angles_limit= {});

  /**
   * @brief  Reset the specified sensor on the device.
   *
   * @param device_id  The ID of the device.
   * @param sensor_id  The ID of the sensor to reset.
   * @return ErrorStruct  Structure indicating success or failure and related
   * error information.
   */
  XHAND_CONTROL_PUBLIC ErrorStruct reset_sensor(const uint8_t device_id,
                                                const uint8_t sensor_id);

  /**
   * @brief  Close the communication channel with the device.
   */
  XHAND_CONTROL_PUBLIC void close_device();

  /**
   * @brief  List the IDs of all connected hands.
   *
   * @return std::vector<uint8_t>  A vector containing the IDs of the connected
   * hands.
   */
  XHAND_CONTROL_PUBLIC std::vector<uint8_t> list_hands_id();

  /**
   * @brief  Get the type of hand for the specified device ID.
   *
   * @param device_id  The ID of the device.
   * @return std::pair<ErrorStruct, std::string>  A pair containing error
   * information and the hand type.
   */
  XHAND_CONTROL_PUBLIC std::pair<ErrorStruct, std::string> get_hand_type(
      uint8_t device_id);

  /**
   * @brief  Get the serial number of the specified device.
   *
   * @param device_id  The ID of the device.
   * @return std::pair<ErrorStruct, std::string>  A pair containing error
   * information and the serial number.
   */
  XHAND_CONTROL_PUBLIC std::pair<ErrorStruct, std::string> get_serial_number(
      uint8_t device_id);

  /**
   * @brief  Set a new name for the specified hand.
   *
   * @param device_id  The ID of the device.
   * @param name  The new name to set for the hand.
   * @return ErrorStruct  Structure indicating success or failure and related
   * error information.
   */
  XHAND_CONTROL_PUBLIC ErrorStruct set_hand_name(const uint8_t device_id,
                                                 const std::string& name);

  /**
   * @brief  Get the name of the specified hand.
   *
   * @param device_id  The ID of the device.
   * @return std::pair<ErrorStruct, std::string>  A pair containing error
   * information and the hand name.
   */
  XHAND_CONTROL_PUBLIC std::pair<ErrorStruct, std::string> get_hand_name(
      uint8_t device_id);

  /**
   * @brief  Change the ID of the hand.
   *
   * @param old_id  The current ID of the hand.
   * @param new_id  The new ID to assign to the hand.
   * @return ErrorStruct  Structure indicating success or failure and related
   * error information.
   */
  XHAND_CONTROL_PUBLIC ErrorStruct set_hand_id(uint8_t old_id, uint8_t new_id);

  /**
   * @brief  Get the SDK version.
   *
   * @return std::string  The SDK version.
   */
  XHAND_CONTROL_PUBLIC std::string get_sdk_version() const;

 private:
  std::shared_ptr<communication_interface::CommunicationInterface> device_;
  HandParam_t hand_parameters_;
  uint8_t calibration_step_{1};
  ErrorStructPtr error_;
};
}  // namespace xhand_control

#endif  // _XHAND_CONTROL_HPP__