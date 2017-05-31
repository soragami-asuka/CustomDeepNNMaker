//======================================
// 畳みこみニューラルネットワークのレイヤーデータ
//======================================
#include"stdafx.h"

#include"UpSampling_LayerData_Base.h"
#include"UpSampling_FUNC.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	/** コンストラクタ */
	UpSampling_LayerData_Base::UpSampling_LayerData_Base(const Gravisbell::GUID& guid)
		: ISingleInputLayerData(), ISingleOutputLayerData()
		,	guid	(guid)
		,	inputDataStruct	()	/**< 入力データ構造 */
		,	pLayerStructure	(NULL)	/**< レイヤー構造を定義したコンフィグクラス */
		,	layerStructure	()		/**< レイヤー構造 */
	{
	}
	/** デストラクタ */
	UpSampling_LayerData_Base::~UpSampling_LayerData_Base()
	{
		if(pLayerStructure != NULL)
			delete pLayerStructure;
	}


	//===========================
	// 共通処理
	//===========================
	/** レイヤー固有のGUIDを取得する */
	Gravisbell::GUID UpSampling_LayerData_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** レイヤー種別識別コードを取得する.
		@param o_layerCode	格納先バッファ
		@return 成功した場合0 */
	Gravisbell::GUID UpSampling_LayerData_Base::GetLayerCode(void)const
	{
		Gravisbell::GUID layerCode;
		::GetLayerCode(layerCode);

		return layerCode;
	}

	//===========================
	// 初期化
	//===========================
	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode UpSampling_LayerData_Base::Initialize(void)
	{
		// 入力バッファ数を確認
		U32 inputBufferCount = this->inputDataStruct.ch * this->inputDataStruct.z * this->inputDataStruct.y * this->inputDataStruct.x;
		if(inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		U32 outputBufferCount = this->outputDataStruct.ch * this->outputDataStruct.z * this->outputDataStruct.y * this->outputDataStruct.x;
		if(outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 初期化. 各ニューロンの値をランダムに初期化
		@param	i_config			設定情報
		@oaram	i_inputDataStruct	入力データ構造情報
		@return	成功した場合0 */
	ErrorCode UpSampling_LayerData_Base::Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct)
	{
		ErrorCode err;

		// 設定情報の登録
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// 入力データ構造の設定
		this->inputDataStruct = i_inputDataStruct;

		// 出力データ構造の設定
		this->outputDataStruct.ch = this->inputDataStruct.ch;
		this->outputDataStruct.x  = this->inputDataStruct.x * this->layerStructure.UpScale.x;
		this->outputDataStruct.y  = this->inputDataStruct.y * this->layerStructure.UpScale.y;
		this->outputDataStruct.z  = this->inputDataStruct.z * this->layerStructure.UpScale.z;

		return this->Initialize();
	}
	/** 初期化. バッファからデータを読み込む
		@param i_lpBuffer	読み込みバッファの先頭アドレス.
		@param i_bufferSize	読み込み可能バッファのサイズ.
		@return	成功した場合0 */
	ErrorCode UpSampling_LayerData_Base::InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize )
	{
		int readBufferByte = 0;

		// 入力データ構造
		IODataStruct inputDataStruct;
		memcpy(&inputDataStruct, &i_lpBuffer[readBufferByte], sizeof(inputDataStruct));
		readBufferByte += sizeof(inputDataStruct);

		// 設定情報
		S32 useBufferByte = 0;
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(&i_lpBuffer[readBufferByte], i_bufferSize, useBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		readBufferByte += useBufferByte;

		// 初期化
		this->Initialize(*pLayerStructure, inputDataStruct);
		delete pLayerStructure;

		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// レイヤー設定
	//===========================
	/** 設定情報を設定 */
	ErrorCode UpSampling_LayerData_Base::SetLayerConfig(const SettingData::Standard::IData& config)
	{
		Gravisbell::ErrorCode err = ERROR_CODE_NONE;

		// レイヤーコードを確認
		{
			Gravisbell::GUID config_guid;
			err = config.GetLayerCode(config_guid);
			if(err != ERROR_CODE_NONE)
				return err;

			Gravisbell::GUID layer_guid;
			err = ::GetLayerCode(layer_guid);
			if(err != ERROR_CODE_NONE)
				return err;

			if(config_guid != layer_guid)
				return ERROR_CODE_INITLAYER_DISAGREE_CONFIG;
		}

		if(this->pLayerStructure != NULL)
			delete this->pLayerStructure;
		this->pLayerStructure = config.Clone();

		// 構造体に読み込む
		this->pLayerStructure->WriteToStruct((BYTE*)&this->layerStructure);

		return ERROR_CODE_NONE;
	}

	/** レイヤーの設定情報を取得する */
	const SettingData::Standard::IData* UpSampling_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
	U32 UpSampling_LayerData_Base::GetUseBufferByteCount()const
	{
		U32 bufferSize = 0;

		if(pLayerStructure == NULL)
			return 0;

		// 入力データ構造
		bufferSize += sizeof(this->inputDataStruct);

		// 設定情報
		bufferSize += pLayerStructure->GetUseBufferByteCount();

		// 本体のバイト数

		return bufferSize;
	}
	
	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S32 UpSampling_LayerData_Base::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		int writeBufferByte = 0;

		// 入力データ構造
		memcpy(&o_lpBuffer[writeBufferByte], &this->inputDataStruct, sizeof(this->inputDataStruct));
		writeBufferByte += sizeof(this->inputDataStruct);

		// 設定情報
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		return writeBufferByte;
	}


	//===========================
	// 入力レイヤー関連
	//===========================
	/** 入力データ構造を取得する.
		@return	入力データ構造 */
	IODataStruct UpSampling_LayerData_Base::GetInputDataStruct()const
	{
		return this->inputDataStruct;
	}

	/** 入力バッファ数を取得する. */
	U32 UpSampling_LayerData_Base::GetInputBufferCount()const
	{
		return this->inputDataStruct.GetDataCount();
	}


	//===========================
	// 出力レイヤー関連
	//===========================
	/** 出力データ構造を取得する */
	IODataStruct UpSampling_LayerData_Base::GetOutputDataStruct()const
	{
		return this->outputDataStruct;
	}

	/** 出力バッファ数を取得する */
	unsigned int UpSampling_LayerData_Base::GetOutputBufferCount()const
	{
		return this->GetOutputDataStruct().GetDataCount();
	}

	
	//===========================
	// 固有関数
	//===========================

} // Gravisbell;
} // Layer;
} // NeuralNetwork;
