/*--------------------------------------------
 * FileName  : FullyConnect_Activation_DATA.hpp
 * LayerName : 全結合ニューラルネットワークレイヤー(活性化関数付き)
 * guid      : BEBA34EC-C30C-4565-9386-56088981D2D7
 * 
 * Text      : 全結合ニューラルネットワークレイヤー.
 *           : 結合層と活性化層を一体化.
 *           : 学習時に[学習係数][ドロップアウト率]を設定できる.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_FUNC_FullyConnect_Activation_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_FUNC_FullyConnect_Activation_H__

#define EXPORT_API extern "C" __declspec(dllexport)

#include<Common/guiddef.h>
#include<Common/ErrorCode.h>
#include<Common/VersionCode.h>

#include<SettingData/Standard/IData.h>
#include<Layer/NeuralNetwork/INNLayer.h>
#include<Layer/NeuralNetwork/ILayerDLLManager.h>

#include"FullyConnect_Activation_DATA.hpp"


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
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLayerStructureSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);


/** Create a learning setting.
  * @return If successful, new configuration information. */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLearningSetting(void);

/** Create learning settings from buffer.
  * @param  i_lpBuffer       Start address of the read buffer.
  * @param  i_bufferSize     The size of the readable buffer.
  * @param  o_useBufferSize  Buffer size actually read.
  * @return If successful, the configuration information created from the buffer
  */
EXPORT_API Gravisbell::SettingData::Standard::IData* CreateLearningSettingFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);


/** Create a layer for CPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayer* CreateLayerCPU(Gravisbell::GUID guid, const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager);

/** Create a layer for GPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayer* CreateLayerGPU(Gravisbell::GUID guid, const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager);



#endif // __GRAVISBELL_NEURAULNETWORK_LAYER_FUNC_FullyConnect_Activation_H__
