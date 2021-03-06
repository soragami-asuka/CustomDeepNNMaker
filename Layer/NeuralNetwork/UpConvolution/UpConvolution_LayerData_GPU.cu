//======================================
// 畳み込みニューラルネットワークのレイヤーデータ
// GPU制御
//======================================
#include"stdafx.h"

#include"UpConvolution_LayerData_GPU.cuh"
#include"UpConvolution_FUNC.hpp"
#include"UpConvolution_GPU.cuh"

#include"RandomUtility.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// コンストラクタ / デストラクタ
	//===========================
	/** コンストラクタ */
	UpConvolution_LayerData_GPU::UpConvolution_LayerData_GPU(const Gravisbell::GUID& guid)
		:	UpConvolution_LayerData_Base(guid)
	{
	}
	/** デストラクタ */
	UpConvolution_LayerData_GPU::~UpConvolution_LayerData_GPU()
	{
	}


	//===========================
	// 初期化
	//===========================
	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode UpConvolution_LayerData_GPU::Initialize(void)
	{
		// 乱数固定化
		Utility::Random::Initialize(0);

		// 入力バッファ数を確認
		U32 inputBufferCount = this->inputDataStruct.ch * this->layerStructure.FilterSize.z * this->layerStructure.FilterSize.y * this->layerStructure.FilterSize.x;
		if(inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// ニューロン数を確認
		U32 neuronCount = this->layerStructure.Output_Channel;
		if(neuronCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;


		// 畳みこみ回数を計算
		this->convolutionCount.x = (S32)ceilf((F32)((this->inputDataStruct.x * this->layerStructure.UpScale.x + this->layerStructure.Padding.x*2 - (this->layerStructure.FilterSize.x - 1)) / this->layerStructure.Stride.x));
		this->convolutionCount.y = (S32)ceilf((F32)((this->inputDataStruct.y * this->layerStructure.UpScale.y + this->layerStructure.Padding.y*2 - (this->layerStructure.FilterSize.y - 1)) / this->layerStructure.Stride.y));
		this->convolutionCount.z = (S32)ceilf((F32)((this->inputDataStruct.z * this->layerStructure.UpScale.z + this->layerStructure.Padding.z*2 - (this->layerStructure.FilterSize.z - 1)) / this->layerStructure.Stride.z));


		// バッファを確保しつつ、初期値を設定
		this->lppNeuron_d.resize(neuronCount * inputBufferCount);
		this->lpBias_d.resize(neuronCount);
		
		thrust::host_vector<F32> lpTmpNeuron(this->lppNeuron_d.size());
		thrust::host_vector<F32> lpTmpBias(this->lpBias_d.size());

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
	ErrorCode UpConvolution_LayerData_GPU::Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct)
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
	ErrorCode UpConvolution_LayerData_GPU::InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize )
	{
		int readBufferByte = 0;

		// 入力データ構造
		memcpy(&this->inputDataStruct, &i_lpBuffer[readBufferByte], sizeof(this->inputDataStruct));
		readBufferByte += sizeof(this->inputDataStruct);

		// 設定情報
		S32 useBufferByte = 0;
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
			thrust::raw_pointer_cast(&this->lppNeuron_d[0]),
			&i_lpBuffer[readBufferByte],
			sizeof(F32) * this->lppNeuron_d.size(),
			cudaMemcpyDeviceToHost);
		readBufferByte += sizeof(F32) * (S32)this->lppNeuron_d.size();

		// バイアス
		cudaMemcpy(
			thrust::raw_pointer_cast(&this->lpBias_d[0]),
			&i_lpBuffer[readBufferByte],
			sizeof(F32) * this->lpBias_d.size(),
			cudaMemcpyDeviceToHost);
		readBufferByte += sizeof(F32) * (S32)this->lpBias_d.size();

		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S32 UpConvolution_LayerData_GPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		int writeBufferByte = 0;

		// 入力データ構造
		memcpy(&o_lpBuffer[writeBufferByte], &this->inputDataStruct, sizeof(this->inputDataStruct));
		writeBufferByte += sizeof(this->inputDataStruct);

		// 設定情報
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		// ニューロン
		cudaMemcpy(
			&o_lpBuffer[writeBufferByte],
			thrust::raw_pointer_cast(&this->lppNeuron_d[0]),
			sizeof(F32) * this->lppNeuron_d.size(),
			cudaMemcpyDeviceToHost);
		writeBufferByte += sizeof(F32) * (S32)this->lppNeuron_d.size();

		// バイアス
		cudaMemcpy(
			&o_lpBuffer[writeBufferByte],
			thrust::raw_pointer_cast(&this->lpBias_d[0]),
			sizeof(F32) * this->lpBias_d.size(),
			cudaMemcpyDeviceToHost);
		writeBufferByte += sizeof(F32) * (S32)this->lpBias_d.size();


		return writeBufferByte;
	}


	//===========================
	// レイヤー作成
	//===========================
	/** レイヤーを作成する.
		@param guid	新規生成するレイヤーのGUID. */
	ILayerBase* UpConvolution_LayerData_GPU::CreateLayer(const Gravisbell::GUID& guid)
	{
		return new UpConvolution_GPU(guid, *this);
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

using namespace Gravisbell;

/** Create a layer for GPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data, const Gravisbell::IODataStruct& i_inputDataStruct)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::UpConvolution_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::UpConvolution_LayerData_GPU(guid);
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
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataGPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::UpConvolution_LayerData_GPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::UpConvolution_LayerData_GPU(guid);
	if(pLayerData == NULL)
		return NULL;

	// 初期化
	S32 useBufferSize = 0;
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
