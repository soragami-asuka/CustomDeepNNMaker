/*--------------------------------------------
 * FileName  : Convolution_DATA.hpp
 * LayerName : ��݂��݃j���[�����l�b�g���[�N
 * guid      : F6662E0E-1CA4-4D59-ACCA-CAC29A16C0AA
 * 
 * Text      : ��݂��݃j���[�����l�b�g���[�N.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_FUNC_Convolution_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_FUNC_Convolution_H__

#define EXPORT_API extern "C" __declspec(dllexport)

#include<Common/common.h>
#include<Common/guiddef.h>
#include<Common/ErrorCode.h>
#include<Common/VersionCode.h>

#include<SettingData/Standard/IData.h>
#include<Layer/NeuralNetwork/INNLayer.h>
#include<Layer/NeuralNetwork/INNLayerData.h>
#include<Layer/NeuralNetwork/ILayerDLLManager.h>

#include"Convolution_DATA.hpp"


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
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayerData* CreateLayerDataCPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data, const Gravisbell::IODataStruct& i_inputDataStruct);
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayerData* CreateLayerDataCPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize );

/** Create a layer for GPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayerData* CreateLayerDataGPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data, const Gravisbell::IODataStruct& i_inputDataStruct);
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayerData* CreateLayerDataGPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize);



#endif // __GRAVISBELL_NEURAULNETWORK_LAYER_FUNC_Convolution_H__