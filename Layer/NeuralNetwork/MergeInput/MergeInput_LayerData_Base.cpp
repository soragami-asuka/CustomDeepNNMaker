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
		:	IMultInputLayerData(), ISingleOutputLayerData()
		,	guid	(guid)
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
		// 出力データ構造を決定する

		this->outputDataStruct = this->lpInputDataStruct[0];

		// 結合後の結合対象軸の要素数を追加
		for(U32 inputNum=1; inputNum<this->lpInputDataStruct.size(); inputNum++)
		{
			switch(this->layerStructure.mergeDirection)
			{
			case MergeInput::LayerStructure::mergeDirection_x:	this->outputDataStruct.x  += this->lpInputDataStruct[inputNum].x;	break;
			case MergeInput::LayerStructure::mergeDirection_y:	this->outputDataStruct.y  += this->lpInputDataStruct[inputNum].y;	break;
			case MergeInput::LayerStructure::mergeDirection_z:	this->outputDataStruct.z  += this->lpInputDataStruct[inputNum].z;	break;
			case MergeInput::LayerStructure::mergeDirection_ch:	this->outputDataStruct.ch += this->lpInputDataStruct[inputNum].ch;	break;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 初期化. 各ニューロンの値をランダムに初期化
		@param	i_config			設定情報
		@oaram	i_inputDataStruct	入力データ構造情報
		@return	成功した場合0 */
	ErrorCode MergeInput_LayerData_Base::Initialize(const SettingData::Standard::IData& i_data, const IODataStruct i_lpInputDataStruct[], U32 i_inputDataCount)
	{
		ErrorCode err;

		// 設定情報の登録
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// 入力データが一つ以上存在することを確認
		if(i_inputDataCount <= 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 入力データ構造の各要素が同じであることを確認
		for(U32 inputNum=1; inputNum<i_inputDataCount; inputNum++)
		{
			if(i_lpInputDataStruct[inputNum-1].x != i_lpInputDataStruct[inputNum].x && this->layerStructure.mergeDirection != MergeInput::LayerStructure::mergeDirection_x)
			{
				return ErrorCode::ERROR_CODE_IO_DISAGREE_INPUT_OUTPUT_COUNT;
			}
			if(i_lpInputDataStruct[inputNum-1].y != i_lpInputDataStruct[inputNum].y && this->layerStructure.mergeDirection != MergeInput::LayerStructure::mergeDirection_y)
			{
				return ErrorCode::ERROR_CODE_IO_DISAGREE_INPUT_OUTPUT_COUNT;
			}
			if(i_lpInputDataStruct[inputNum-1].z != i_lpInputDataStruct[inputNum].z && this->layerStructure.mergeDirection != MergeInput::LayerStructure::mergeDirection_z)
			{
				return ErrorCode::ERROR_CODE_IO_DISAGREE_INPUT_OUTPUT_COUNT;
			}
			if(i_lpInputDataStruct[inputNum-1].ch != i_lpInputDataStruct[inputNum].ch && this->layerStructure.mergeDirection != MergeInput::LayerStructure::mergeDirection_ch)
			{
				return ErrorCode::ERROR_CODE_IO_DISAGREE_INPUT_OUTPUT_COUNT;
			}
		}

		// 入力データ構造の設定
		this->lpInputDataStruct.resize(i_inputDataCount);
		for(U32 inputNum=0; inputNum<i_inputDataCount; inputNum++)
		{
			this->lpInputDataStruct[inputNum] = i_lpInputDataStruct[inputNum];
		}

		return this->Initialize();
	}
	/** 初期化. バッファからデータを読み込む
		@param i_lpBuffer	読み込みバッファの先頭アドレス.
		@param i_bufferSize	読み込み可能バッファのサイズ.
		@return	成功した場合0 */
	ErrorCode MergeInput_LayerData_Base::InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize)
	{
		int readBufferByte = 0;

		// 入力データ数
		U32 inputDataCount = 0;
		memcpy(&inputDataCount, i_lpBuffer, sizeof(U32));
		// 入力データ構造
		std::vector<IODataStruct> lpTmpInputDataStruct(inputDataCount);
		for(U32 inputNum=0; inputNum<inputDataCount; inputNum++)
		{
			memcpy(&lpTmpInputDataStruct[inputNum], &i_lpBuffer[readBufferByte], sizeof(IODataStruct));
			readBufferByte += sizeof(IODataStruct);
		}

		// 設定情報
		S32 useBufferByte = 0;
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(&i_lpBuffer[readBufferByte], i_bufferSize, useBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;

		// 初期化する
		ErrorCode err = this->Initialize(*pLayerStructure, &lpInputDataStruct[0], (U32)lpInputDataStruct.size());
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

		// 入力データ数
		bufferSize += sizeof(U32);

		// 入力データ構造
		bufferSize += sizeof(IODataStruct) * (U32)this->lpInputDataStruct.size();

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

		// 入力データ数
		U32 inputDataCount = (U32)this->lpInputDataStruct.size();
		memcpy(&o_lpBuffer[writeBufferByte], &inputDataCount, sizeof(U32));
		writeBufferByte += sizeof(U32);

		// 入力データ構造
		for(U32 inputNum=0; inputNum<this->lpInputDataStruct.size(); inputNum++)
		{
			memcpy(&o_lpBuffer[writeBufferByte], &this->lpInputDataStruct[inputNum], sizeof(IODataStruct));
			writeBufferByte += sizeof(IODataStruct);
		}

		// 設定情報
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		return writeBufferByte;
	}



	//===========================
	// 入力レイヤー関連
	//===========================
	/** 入力データの数を取得する */
	U32 MergeInput_LayerData_Base::GetInputDataCount()const
	{
		return (U32)this->lpInputDataStruct.size();
	}

	/** 入力データ構造を取得する.
		@return	入力データ構造 */
	IODataStruct MergeInput_LayerData_Base::GetInputDataStruct(U32 i_dataNum)const
	{
		if(i_dataNum >= this->lpInputDataStruct.size())
			return IODataStruct(0, 0, 0, 0);

		return this->lpInputDataStruct[i_dataNum];
	}

	/** 入力バッファ数を取得する. */
	U32 MergeInput_LayerData_Base::GetInputBufferCount(U32 i_dataNum)const
	{
		return this->GetInputDataStruct(i_dataNum).GetDataCount();
	}


	//===========================
	// 出力レイヤー関連
	//===========================
	/** 出力データ構造を取得する */
	IODataStruct MergeInput_LayerData_Base::GetOutputDataStruct()const
	{
		return this->outputDataStruct;
	}

	/** 出力バッファ数を取得する */
	unsigned int MergeInput_LayerData_Base::GetOutputBufferCount()const
	{
		return GetOutputDataStruct().GetDataCount();
	}

	
	//===========================
	// 固有関数
	//===========================

} // Gravisbell;
} // Layer;
} // NeuralNetwork;
