//======================================
// バッチ正規化のレイヤーデータ
// CPU制御
//======================================
#include"stdafx.h"

#include"BatchNormalization_LayerData_CPU.h"
#include"BatchNormalization_FUNC.hpp"
#include"BatchNormalization_CPU.h"

using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	//===========================
	// コンストラクタ / デストラクタ
	//===========================
	/** コンストラクタ */
	BatchNormalization_LayerData_CPU::BatchNormalization_LayerData_CPU(const Gravisbell::GUID& guid)
		:	BatchNormalization_LayerData_Base(guid)
	{
	}
	/** デストラクタ */
	BatchNormalization_LayerData_CPU::~BatchNormalization_LayerData_CPU()
	{
	}


	//===========================
	// 初期化
	//===========================
	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode BatchNormalization_LayerData_CPU::Initialize(void)
	{
		this->lpMean.resize(this->inputDataStruct.ch);
		this->lpVariance.resize(this->inputDataStruct.ch);
		this->lpScale.resize(this->inputDataStruct.ch);
		this->lpBias.resize(this->inputDataStruct.ch);

		for(U32 ch=0; ch<this->inputDataStruct.ch; ch++)
		{
			this->lpMean[ch] = 0.0f;
			this->lpVariance[ch] = 0.0f;
			this->lpScale[ch] = 1.0f;
			this->lpBias[ch] = 0.0f;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 初期化. 各ニューロンの値をランダムに初期化
		@param	i_config			設定情報
		@oaram	i_inputDataStruct	入力データ構造情報
		@return	成功した場合0 */
	ErrorCode BatchNormalization_LayerData_CPU::Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct)
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
	ErrorCode BatchNormalization_LayerData_CPU::InitializeFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize)
	{
		int readBufferByte = 0;

		// 入力データ構造
		memcpy(&this->inputDataStruct, &i_lpBuffer[readBufferByte], sizeof(this->inputDataStruct));
		readBufferByte += sizeof(this->inputDataStruct);

		// 設定情報
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(&i_lpBuffer[readBufferByte], i_bufferSize, readBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		this->SetLayerConfig(*pLayerStructure);
		delete pLayerStructure;

		// 初期化する
		this->Initialize();

		// 平均
		memcpy(&this->lpMean[0], &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size());
		// 分散
		memcpy(&this->lpVariance[0], &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size());
		// スケーリング値
		memcpy(&this->lpScale[0], &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size());
		// バイアス値
		memcpy(&this->lpBias[0], &i_lpBuffer[readBufferByte], sizeof(F32)*this->lpMean.size());


		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S32 BatchNormalization_LayerData_CPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		int writeBufferByte = 0;

		// 入力データ構造
		memcpy(&o_lpBuffer[writeBufferByte], &this->inputDataStruct, sizeof(this->inputDataStruct));
		writeBufferByte += sizeof(this->inputDataStruct);

		// 設定情報
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);


		// 平均
		memcpy(&o_lpBuffer[writeBufferByte], &this->lpMean[0], sizeof(F32)*this->lpMean.size());
		writeBufferByte += sizeof(F32)*this->lpMean.size();
		// 分散
		memcpy(&o_lpBuffer[writeBufferByte], &this->lpVariance[0], sizeof(F32)*this->lpVariance.size());
		writeBufferByte += sizeof(F32)*this->lpVariance.size();
		// スケーリング値
		memcpy(&o_lpBuffer[writeBufferByte], &this->lpScale[0], sizeof(F32)*this->lpScale.size());
		writeBufferByte += sizeof(F32)*this->lpScale.size();
		// バイアス値
		memcpy(&o_lpBuffer[writeBufferByte], &this->lpBias[0], sizeof(F32)*this->lpBias.size());
		writeBufferByte += sizeof(F32)*this->lpBias.size();


		return writeBufferByte;
	}


	//===========================
	// レイヤー作成
	//===========================
	/** レイヤーを作成する.
		@param guid	新規生成するレイヤーのGUID. */
	INNLayer* BatchNormalization_LayerData_CPU::CreateLayer(const Gravisbell::GUID& guid)
	{
		return new BatchNormalization_CPU(guid, *this);
	}

} // Gravisbell;
} // Layer;
} // NeuralNetwork;


/** Create a layer for CPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayerData* CreateLayerDataCPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data, const Gravisbell::IODataStruct& i_inputDataStruct)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::BatchNormalization_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::BatchNormalization_LayerData_CPU(guid);
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
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayerData* CreateLayerDataCPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)
{
	// 作成
	Gravisbell::Layer::NeuralNetwork::BatchNormalization_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::BatchNormalization_LayerData_CPU(guid);
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
