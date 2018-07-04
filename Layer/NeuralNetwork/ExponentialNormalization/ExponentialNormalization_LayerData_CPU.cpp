//======================================
// バッチ正規化のレイヤーデータ
// CPU制御
//======================================
#include"stdafx.h"

#include"ExponentialNormalization_LayerData_CPU.h"
#include"ExponentialNormalization_FUNC.hpp"
#include"ExponentialNormalization_CPU.h"

#include"Library/NeuralNetwork/Optimizer.h"

#include"../_LayerBase/CLayerBase_CPU.h"

using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// コンストラクタ / デストラクタ
	//===========================
	/** コンストラクタ */
	ExponentialNormalization_LayerData_CPU::ExponentialNormalization_LayerData_CPU(const Gravisbell::GUID& guid)
		:	ExponentialNormalization_LayerData_Base(guid)
	{
	}
	/** デストラクタ */
	ExponentialNormalization_LayerData_CPU::~ExponentialNormalization_LayerData_CPU()
	{
	}


	//===========================
	// 初期化
	//===========================
	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode ExponentialNormalization_LayerData_CPU::Initialize(void)
	{
		this->lpMean.resize(this->layerStructure.InputChannelCount);
		this->lpVariance.resize(this->layerStructure.InputChannelCount);

		for(U32 ch=0; ch<this->layerStructure.InputChannelCount; ch++)
		{
			this->lpMean[ch] = 0.0f;
			this->lpVariance[ch] = 1.0f;
		}

		this->learnTime = 0;	/**< 学習回数 */


		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 初期化. 各ニューロンの値をランダムに初期化
		@param	i_config			設定情報
		@oaram	i_inputDataStruct	入力データ構造情報
		@return	成功した場合0 */
	ErrorCode ExponentialNormalization_LayerData_CPU::Initialize(const SettingData::Standard::IData& i_data)
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

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 初期化. バッファからデータを読み込む
		@param i_lpBuffer	読み込みバッファの先頭アドレス.
		@param i_bufferSize	読み込み可能バッファのサイズ.
		@return	成功した場合0 */
	ErrorCode ExponentialNormalization_LayerData_CPU::InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize, S64& o_useBufferSize)
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

		// 平均
		memcpy(&this->lpMean[0], &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size());
		readBufferByte += sizeof(F32)*(S32)this->lpMean.size();
		// 分散
		memcpy(&this->lpVariance[0], &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size());
		readBufferByte += sizeof(F32)*(S32)this->lpMean.size();

		// 学習回数
		memcpy(&this->learnTime, &i_lpBuffer[readBufferByte], sizeof(this->learnTime));
		readBufferByte += sizeof(this->learnTime);

		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S64 ExponentialNormalization_LayerData_CPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		S64 writeBufferByte = 0;

		// 設定情報
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		// 平均
		memcpy(&o_lpBuffer[writeBufferByte], &this->lpMean[0], sizeof(F32)*this->lpMean.size());
		writeBufferByte += sizeof(F32)*(S32)this->lpMean.size();
		// 分散
		memcpy(&o_lpBuffer[writeBufferByte], &this->lpVariance[0], sizeof(F32)*this->lpVariance.size());
		writeBufferByte += sizeof(F32)*(S32)this->lpVariance.size();

		// 学習回数
		memcpy(&o_lpBuffer[writeBufferByte], &this->learnTime, sizeof(this->learnTime));
		writeBufferByte += sizeof(U64);


		return writeBufferByte;
	}


	//===========================
	// レイヤー作成
	//===========================
	/** レイヤーを作成する.
		@param guid	新規生成するレイヤーのGUID. */
	ILayerBase* ExponentialNormalization_LayerData_CPU::CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return NULL;

		return new CNNSingle2SingleLayerBase_CPU<ExponentialNormalization_CPU, ExponentialNormalization_LayerData_CPU>(guid, *this, i_lpInputDataStruct[0], i_temporaryMemoryManager);
	}


	//===========================
	// オプティマイザー設定
	//===========================		
	/** オプティマイザーを変更する */
	ErrorCode ExponentialNormalization_LayerData_CPU::ChangeOptimizer(const wchar_t i_optimizerID[])
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;


/** Create a layer for CPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::ExponentialNormalization_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::ExponentialNormalization_LayerData_CPU(guid);
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
EXPORT_API Gravisbell::Layer::ILayerData* CreateLayerDataCPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S64 i_bufferSize, S64& o_useBufferSize)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::ExponentialNormalization_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::ExponentialNormalization_LayerData_CPU(guid);
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
