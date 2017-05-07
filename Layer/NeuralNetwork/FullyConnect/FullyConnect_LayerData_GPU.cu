//======================================
// 全結合ニューラルネットワークのレイヤーデータ
// GPU制御
//======================================
#include"stdafx.h"

#include"FullyConnect_LayerData_GPU.cuh"
#include"FullyConnect_FUNC.hpp"
#include"FullyConnect_GPU.cuh"

#include"RandomUtility.h"

#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// コンストラクタ / デストラクタ
	//===========================
	/** コンストラクタ */
	FullyConnect_LayerData_GPU::FullyConnect_LayerData_GPU(const Gravisbell::GUID& guid)
		:	FullyConnect_LayerData_Base(guid)
	{
	}
	/** デストラクタ */
	FullyConnect_LayerData_GPU::~FullyConnect_LayerData_GPU()
	{
	}


	//===========================
	// 初期化
	//===========================
	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode FullyConnect_LayerData_GPU::Initialize(void)
	{
		// 入力バッファ数を確認
		unsigned int inputBufferCount = this->GetInputBufferCount();
		if(inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// ニューロン数を確認
		unsigned int neuronCount = this->GetNeuronCount();
		if(neuronCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// バッファを確保しつつ、初期値を設定
		this->lppNeuron_d.reserve(neuronCount * inputBufferCount);
		this->lppNeuron_d.resize(neuronCount * inputBufferCount);
		this->lpBias_d.reserve(neuronCount);
		this->lpBias_d.resize(neuronCount);

		thrust::host_vector<F32> lpTmpNeuron(neuronCount * inputBufferCount);
		thrust::host_vector<F32> lpTmpBias(neuronCount);

		float maxArea = sqrt(6.0f / (inputBufferCount + neuronCount));
		for(U32 i=0; i<lpTmpNeuron.size(); i++)
			lpTmpNeuron[i] = ((F32)Utility::Random::GetValue() - 0.5f) * 2.0f * maxArea;
		for(U32 i=0; i<lpTmpBias.size(); i++)
			lpTmpBias[i] = ((F32)Utility::Random::GetValue() - 0.5f) * 2.0f * maxArea;

		this->lppNeuron_d = lpTmpNeuron;
		this->lpBias_d = lpTmpBias;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 初期化. 各ニューロンの値をランダムに初期化
		@param	i_config			設定情報
		@oaram	i_inputDataStruct	入力データ構造情報
		@return	成功した場合0 */
	ErrorCode FullyConnect_LayerData_GPU::Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct)
	{
		ErrorCode err;

		// 設定情報の登録
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// 入力データ構造の設定
		this->inputDataStruct = i_inputDataStruct;

		return this->Initialize();
	}
	/** 初期化. バッファからデータを読み込む
		@param i_lpBuffer	読み込みバッファの先頭アドレス.
		@param i_bufferSize	読み込み可能バッファのサイズ.
		@return	成功した場合0 */
	ErrorCode FullyConnect_LayerData_GPU::InitializeFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize)
	{
		int readBufferByte = 0;

		// 設定情報を読み込む
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(i_lpBuffer, i_bufferSize, readBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		this->SetLayerConfig(*pLayerStructure);
		delete pLayerStructure;

		// 初期化する
		this->Initialize();

		//// ニューロン係数
		//for(unsigned int neuronNum=0; neuronNum<this->lppNeuron.size(); neuronNum++)
		//{
		//	memcpy(&this->lppNeuron[neuronNum][0], &i_lpBuffer[readBufferByte], this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE));
		//	readBufferByte += this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE);
		//}

		//// バイアス
		//memcpy(&this->lpBias[0], &i_lpBuffer[readBufferByte], this->lpBias.size() * sizeof(NEURON_TYPE));
		//readBufferByte += this->lpBias.size() * sizeof(NEURON_TYPE);


		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S32 FullyConnect_LayerData_GPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		int writeBufferByte = 0;

		// 設定情報
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		//// ニューロン係数
		//for(unsigned int neuronNum=0; neuronNum<this->lppNeuron.size(); neuronNum++)
		//{
		//	memcpy(&o_lpBuffer[writeBufferByte], &this->lppNeuron[neuronNum][0], this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE));
		//	writeBufferByte += this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE);
		//}

		//// バイアス
		//memcpy(&o_lpBuffer[writeBufferByte], &this->lpBias[0], this->lpBias.size() * sizeof(NEURON_TYPE));
		//writeBufferByte += this->lpBias.size() * sizeof(NEURON_TYPE);

		return writeBufferByte;
	}


	//===========================
	// レイヤー作成
	//===========================
	/** レイヤーを作成する.
		@param guid	新規生成するレイヤーのGUID. */
	INNLayer* FullyConnect_LayerData_GPU::CreateLayer(const Gravisbell::GUID& guid)
	{
		return new FullyConnect_GPU(guid, *this);
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;


using namespace Gravisbell;

/** Create a layer for GPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayerData* CreateLayerDataGPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data, const Gravisbell::IODataStruct& i_inputDataStruct)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::FullyConnect_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::FullyConnect_LayerData_GPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// 初期化
	Gravisbell::ErrorCode errCode = pLayerData->Initialize(i_data, i_inputDataStruct);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	return pLayerData;
}
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayerData* CreateLayerDataGPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::FullyConnect_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::FullyConnect_LayerData_GPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// 読み取りに使用するバッファ数を取得
	U32 useBufferSize = pLayerData->GetUseBufferByteCount();
	if(useBufferSize >= (U32)i_bufferSize)
	{
		delete pLayerData;
		return NULL;
	}

	// 初期化
	Gravisbell::ErrorCode errCode = pLayerData->InitializeFromBuffer(i_lpBuffer, i_bufferSize);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	// 使用したバッファ量を格納
	o_useBufferSize = useBufferSize;

	return pLayerData;
}