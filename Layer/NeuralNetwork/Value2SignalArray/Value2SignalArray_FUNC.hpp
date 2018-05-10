/*--------------------------------------------
 * FileName  : Value2SignalArray_DATA.hpp
 * LayerName : 信号の配列から値へ変換
 * guid      : 6F6C75B8-9C41-43EA-8F80-98C6F1CF4A2D
 * 
 * Text      : 信号の配列から値へ変換する.
 *           : 最大値を取るCH番号を値に変換する.
 *           : 入力CH数＝分解能の整数倍である必要がある.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_FUNC_Value2SignalArray_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_FUNC_Value2SignalArray_H__

#define EXPORT_API extern "C" __declspec(dllexport)

#include<Common/common.h>
#include<Common/guiddef.h>
#include<Common/ErrorCode.h>
#include<Common/VersionCode.h>

#include<SettingData/Standard/IData.h>
#include<Layer/NeuralNetwork/ILayerDLLManager.h>

#include"Value2SignalArray_DATA.hpp"


/** Acquire the layer identification code.
  * @param  o_layerCode    Storage destination buffer.
  * @return On success 0. 
  */
EXPORT_API Gravisbell::ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode);

/** Get version code.
  * @param  o_versionCode    Storage destination buffer.
  * @return On success 0. 
  */
EXPORT_API Gravisbell::ErrorCode GetVersionCode(Gravisbell::VersionCode& o_versionCode);


/** Create a layer structure setting.
  * @return If successful, new configuration information.
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSetting(void);

/** Create layer structure settings from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize);


/** Create a runtime parameter.
  * @return If successful, new configuration information. */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateRuntimeParameter(void);

/** Create runtime parameter from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateRuntimeParameterFromBuffer(const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize);


/** Create a layer for CPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data);
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize );

/** Create a layer for GPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data);
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, Gravisbell::S64 i_bufferSize, Gravisbell::S64& o_useBufferSize);



#endif // __GRAVISBELL_NEURAULNETWORK_LAYER_FUNC_Value2SignalArray_H__
