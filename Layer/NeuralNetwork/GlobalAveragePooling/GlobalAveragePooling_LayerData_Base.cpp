//======================================
// プーリングレイヤーのデータ
//======================================
#include"stdafx.h"

#include"GlobalAveragePooling_LayerData_Base.h"
#include"GlobalAveragePooling_FUNC.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	GlobalAveragePooling_LayerData_Base::GlobalAveragePooling_LayerData_Base(const Gravisbell::GUID& guid)
		:	guid	(guid)
		,	pLayerStructure	(NULL)	/**< レイヤー構造を定義したコンフィグクラス */
		//,	layerStructure	()		/**< レイヤー構造 */
	{
	}
	/** デストラクタ */
	GlobalAveragePooling_LayerData_Base::~GlobalAveragePooling_LayerData_Base()
	{
		if(pLayerStructure != NULL)
			delete pLayerStructure;
	}


	//===========================
	// 共通処理
	//===========================
	/** レイヤー固有のGUIDを取得する */
	Gravisbell::GUID GlobalAveragePooling_LayerData_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** レイヤー種別識別コードを取得する.
		@param o_layerCode	格納先バッファ
		@return 成功した場合0 */
	Gravisbell::GUID GlobalAveragePooling_LayerData_Base::GetLayerCode(void)const
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
	ErrorCode GlobalAveragePooling_LayerData_Base::Initialize(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 初期化. 各ニューロンの値をランダムに初期化
		@param	i_config			設定情報
		@oaram	i_inputDataStruct	入力データ構造情報
		@return	成功した場合0 */
	ErrorCode GlobalAveragePooling_LayerData_Base::Initialize(const SettingData::Standard::IData& i_data)
	{
		ErrorCode err;

		// 設定情報の登録
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		return this->Initialize();
	}
	/** 初期化. バッファからデータを読み込む
		@param i_lpBuffer	読み込みバッファの先頭アドレス.
		@param i_bufferSize	読み込み可能バッファのサイズ.
		@return	成功した場合0 */
	ErrorCode GlobalAveragePooling_LayerData_Base::InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize)
	{
		int readBufferByte = 0;

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

		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// レイヤー設定
	//===========================
	/** 設定情報を設定 */
	ErrorCode GlobalAveragePooling_LayerData_Base::SetLayerConfig(const SettingData::Standard::IData& config)
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
		//this->pLayerStructure->WriteToStruct((BYTE*)&this->layerStructure);

		return ERROR_CODE_NONE;
	}

	/** レイヤーの設定情報を取得する */
	const SettingData::Standard::IData* GlobalAveragePooling_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
	U32 GlobalAveragePooling_LayerData_Base::GetUseBufferByteCount()const
	{
		U32 bufferSize = 0;

		if(pLayerStructure == NULL)
			return 0;

		// 設定情報
		bufferSize += pLayerStructure->GetUseBufferByteCount();


		return bufferSize;
	}

	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S32 GlobalAveragePooling_LayerData_Base::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		int writeBufferByte = 0;

		// 設定情報
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		return writeBufferByte;
	}



	//===========================
	// レイヤー構造
	//===========================
	/** 入力データ構造が使用可能か確認する.
		@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
		@return	使用可能な入力データ構造の場合trueが返る. */
	bool GlobalAveragePooling_LayerData_Base::CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(i_inputLayerCount == 0)
			return false;
		if(i_inputLayerCount > 1)
			return false;
		if(i_lpInputDataStruct == NULL)
			return false;
		if(i_lpInputDataStruct[0].x == 0)
			return false;
		if(i_lpInputDataStruct[0].y == 0)
			return false;
		if(i_lpInputDataStruct[0].z == 0)
			return false;
		if(i_lpInputDataStruct[0].ch == 0)
			return false;

		return true;
	}

	/** 出力データ構造を取得する.
		@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
		@return	入力データ構造が不正な場合(x=0,y=0,z=0,ch=0)が返る. */
	IODataStruct GlobalAveragePooling_LayerData_Base::GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return IODataStruct(0,0,0,0);

		return IODataStruct(i_lpInputDataStruct[0].ch, 1, 1, 1);
	}

	/** 複数出力が可能かを確認する */
	bool GlobalAveragePooling_LayerData_Base::CheckCanHaveMultOutputLayer(void)
	{
		return false;
	}

	
	//===========================
	// 固有関数
	//===========================

} // Gravisbell;
} // Layer;
} // NeuralNetwork;
