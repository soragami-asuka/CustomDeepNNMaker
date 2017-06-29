//======================================
// プーリングレイヤーのデータ
//======================================
#include"stdafx.h"

#include"MergeInput_LayerData_Base.h"
#include"MergeInput_FUNC.hpp"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	MergeInput_LayerData_Base::MergeInput_LayerData_Base(const Gravisbell::GUID& guid)
		:	guid	(guid)
		,	pLayerStructure	(NULL)	/**< レイヤー構造を定義したコンフィグクラス */
		,	layerStructure	()		/**< レイヤー構造 */
	{
	}
	/** デストラクタ */
	MergeInput_LayerData_Base::~MergeInput_LayerData_Base()
	{
		if(pLayerStructure != NULL)
			delete pLayerStructure;
	}


	//===========================
	// 共通処理
	//===========================
	/** レイヤー固有のGUIDを取得する */
	Gravisbell::GUID MergeInput_LayerData_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** レイヤー種別識別コードを取得する.
		@param o_layerCode	格納先バッファ
		@return 成功した場合0 */
	Gravisbell::GUID MergeInput_LayerData_Base::GetLayerCode(void)const
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
	ErrorCode MergeInput_LayerData_Base::Initialize(void)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 初期化. 各ニューロンの値をランダムに初期化
		@param	i_config			設定情報
		@oaram	i_inputDataStruct	入力データ構造情報
		@return	成功した場合0 */
	ErrorCode MergeInput_LayerData_Base::Initialize(const SettingData::Standard::IData& i_data)
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
	ErrorCode MergeInput_LayerData_Base::InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize)
	{
		int readBufferByte = 0;

		// 設定情報
		S32 useBufferByte = 0;
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(&i_lpBuffer[readBufferByte], i_bufferSize, useBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		readBufferByte += useBufferByte;

		// 初期化する
		ErrorCode err = this->Initialize(*pLayerStructure);
		delete pLayerStructure;
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// レイヤー設定
	//===========================
	/** 設定情報を設定 */
	ErrorCode MergeInput_LayerData_Base::SetLayerConfig(const SettingData::Standard::IData& config)
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
	const SettingData::Standard::IData* MergeInput_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
	U32 MergeInput_LayerData_Base::GetUseBufferByteCount()const
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
	S32 MergeInput_LayerData_Base::WriteToBuffer(BYTE* o_lpBuffer)const
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
	bool MergeInput_LayerData_Base::CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(i_inputLayerCount == 0)
			return false;
		if(i_lpInputDataStruct == NULL)
			return false;

		IODataStruct outputDataStruct = i_lpInputDataStruct[0];
		for(U32 inputNum=1; inputNum<i_inputLayerCount; inputNum++)
		{
			if(this->layerStructure.mergeDirection != MergeInput::LayerStructure::mergeDirection_x  && outputDataStruct.x  != i_lpInputDataStruct[inputNum].x)	return false;
			if(this->layerStructure.mergeDirection != MergeInput::LayerStructure::mergeDirection_y  && outputDataStruct.y  != i_lpInputDataStruct[inputNum].y)	return false;
			if(this->layerStructure.mergeDirection != MergeInput::LayerStructure::mergeDirection_z  && outputDataStruct.z  != i_lpInputDataStruct[inputNum].z)	return false;
			if(this->layerStructure.mergeDirection != MergeInput::LayerStructure::mergeDirection_ch && outputDataStruct.ch != i_lpInputDataStruct[inputNum].ch)	return false;
		}

		return true;
	}

	/** 出力データ構造を取得する.
		@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
		@return	入力データ構造が不正な場合(x=0,y=0,z=0,ch=0)が返る. */
	IODataStruct MergeInput_LayerData_Base::GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(this->CheckCanUseInputDataStruct(i_lpInputDataStruct, i_inputLayerCount) == false)
			return IODataStruct(0,0,0,0);
		
		IODataStruct outputDataStruct = i_lpInputDataStruct[0];

		// 結合後の結合対象軸の要素数を追加
		for(U32 inputNum=1; inputNum<i_inputLayerCount; inputNum++)
		{
			switch(this->layerStructure.mergeDirection)
			{
			case MergeInput::LayerStructure::mergeDirection_x:	outputDataStruct.x  += i_lpInputDataStruct[inputNum].x;		break;
			case MergeInput::LayerStructure::mergeDirection_y:	outputDataStruct.y  += i_lpInputDataStruct[inputNum].y;		break;
			case MergeInput::LayerStructure::mergeDirection_z:	outputDataStruct.z  += i_lpInputDataStruct[inputNum].z;		break;
			case MergeInput::LayerStructure::mergeDirection_ch:	outputDataStruct.ch += i_lpInputDataStruct[inputNum].ch;	break;
			}
		}

		return outputDataStruct;
	}

	/** 複数出力が可能かを確認する */
	bool MergeInput_LayerData_Base::CheckCanHaveMultOutputLayer(void)
	{
		return false;
	}


	//===========================
	// 固有関数
	//===========================

} // Gravisbell;
} // Layer;
} // NeuralNetwork;
