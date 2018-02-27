//======================================
// 全結合ニューラルネットワークのレイヤーデータ
// GPU制御
//======================================
#include"stdafx.h"

#include"SOM_LayerData_GPU.cuh"
#include"SOM_FUNC.hpp"
#include"SOM_GPU.cuh"

#include"../_LayerBase/CLayerBase_GPU.cuh"

#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)

#include"Library/NeuralNetwork/Optimizer.h"
#include"Library/NeuralNetwork/Initializer.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// コンストラクタ / デストラクタ
	//===========================
	/** コンストラクタ */
	SOM_LayerData_GPU::SOM_LayerData_GPU(const Gravisbell::GUID& guid)
		:	SOM_LayerData_Base(guid)
	{
	}
	/** デストラクタ */
	SOM_LayerData_GPU::~SOM_LayerData_GPU()
	{
	}


	//===========================
	// 初期化
	//===========================
	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode SOM_LayerData_GPU::Initialize(void)
	{
		// 入力バッファ数を確認
		unsigned int inputBufferCount = this->GetInputBufferCount();
		if(inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// ニューロン数を確認
		unsigned int unitCount = this->GetUnitCount();
		if(unitCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// バッファを確保しつつ、初期値を設定
		U32 inputCount  = inputBufferCount;
		U32 outputCount = this->layerStructure.DimensionCount;

		this->lpUnitData.resize(unitCount * inputBufferCount);

		thrust::host_vector<F32> lpTmpUnitData(unitCount * inputBufferCount);

		for(U32 i=0; i<lpTmpUnitData.size(); i++)
		{
			lpTmpUnitData[i] = Gravisbell::Layer::NeuralNetwork::GetInitializerManager().GetRandomValue(this->layerStructure.InitializeMinValue, this->layerStructure.InitializeMaxValue);
		}

		this->lpUnitData = lpTmpUnitData;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 初期化. 各ニューロンの値をランダムに初期化
		@param	i_config			設定情報
		@oaram	i_inputDataStruct	入力データ構造情報
		@return	成功した場合0 */
	ErrorCode SOM_LayerData_GPU::Initialize(const SettingData::Standard::IData& i_data)
	{
		ErrorCode err;

		// 設定情報の登録
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// 初期化
		err = this->Initialize();
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// オプティマイザーの設定
		err = this->ChangeOptimizer(L"SGD");
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		return this->Initialize();
	}
	/** 初期化. バッファからデータを読み込む
		@param i_lpBuffer	読み込みバッファの先頭アドレス.
		@param i_bufferSize	読み込み可能バッファのサイズ.
		@return	成功した場合0 */
	ErrorCode SOM_LayerData_GPU::InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize, S64& o_useBufferSize )
	{
		S64 readBufferByte = 0;

		// 設定情報
		S64 useBufferByte = 0;
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(&i_lpBuffer[readBufferByte], i_bufferSize, useBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		readBufferByte += useBufferByte;
		this->SetLayerConfig(*pLayerStructure);
		delete pLayerStructure;

		// 初期化する
		this->Initialize();

		// バッファからコピー
		// ニューロン
		cudaMemcpy(
			thrust::raw_pointer_cast(&this->lpUnitData[0]),
			&i_lpBuffer[readBufferByte],
			sizeof(F32) * this->lpUnitData.size(),
			cudaMemcpyHostToDevice);
		readBufferByte += sizeof(F32) * (S32)this->lpUnitData.size();


		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S64 SOM_LayerData_GPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		S64 writeBufferByte = 0;

		// 設定情報
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		// ニューロン
		cudaMemcpy(
			&o_lpBuffer[writeBufferByte],
			thrust::raw_pointer_cast(&this->lpUnitData[0]),
			sizeof(F32) * this->lpUnitData.size(),
			cudaMemcpyDeviceToHost);
		writeBufferByte += sizeof(F32) * (S32)this->lpUnitData.size();


		return writeBufferByte;
	}


	//===========================
	// レイヤー作成
	//===========================
	/** レイヤーを作成する.
		@param guid	新規生成するレイヤーのGUID. */
	ILayerBase* SOM_LayerData_GPU::CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return NULL;

		return new CNNSingle2SingleLayerBase_GPU<SOM_GPU, SOM_LayerData_GPU>(guid, *this, i_lpInputDataStruct[0], i_temporaryMemoryManager);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;


using namespace Gravisbell;

/** Create a layer for GPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::SOM_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::SOM_LayerData_GPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// 初期化
	Gravisbell::ErrorCode errCode = pLayerData->Initialize(i_data);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	return pLayerData;
}
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S64 i_bufferSize, S64& o_useBufferSize)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::SOM_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::SOM_LayerData_GPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// 初期化
	S64 useBufferSize = 0;
	Gravisbell::ErrorCode errCode = pLayerData->InitializeFromBuffer(i_lpBuffer, i_bufferSize, useBufferSize);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	// 使用したバッファ量を格納
	o_useBufferSize = useBufferSize;

	return pLayerData;
}