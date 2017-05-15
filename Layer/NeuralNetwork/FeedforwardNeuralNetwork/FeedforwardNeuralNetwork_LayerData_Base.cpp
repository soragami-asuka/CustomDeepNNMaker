//======================================
// フィードフォワードニューラルネットワークの処理レイヤーのデータ
// 複数のレイヤーを内包し、処理する
//======================================
#include"stdafx.h"

#include"FeedforwardNeuralNetwork_LayerData_Base.h"
#include"FeedforwardNeuralNetwork_FUNC.hpp"
#include"FeedforwardNeuralNetwork_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	
	//====================================
	// コンストラクタ/デストラクタ
	//====================================
	/** コンストラクタ */
	FeedforwardNeuralNetwork_LayerData_Base::FeedforwardNeuralNetwork_LayerData_Base(const ILayerDLLManager& i_layerDLLManager, const Gravisbell::GUID& guid)
		:	layerDLLManager	(i_layerDLLManager)
		,	guid			(guid)
		,	inputLayerGUID	(Gravisbell::GUID(0x2d2805a3, 0x97cc, 0x4ab4, 0x94, 0x2e, 0x69, 0x39, 0xfd, 0x62, 0x35, 0xb1))
		,	pLayerStructure	(NULL)
	{
	}
	/** デストラクタ */
	FeedforwardNeuralNetwork_LayerData_Base::~FeedforwardNeuralNetwork_LayerData_Base()
	{
		// 内部保有のレイヤーデータを削除
		for(auto it : this->lpLayerData)
			delete it.second;

		// レイヤー構造を削除
		if(this->pLayerStructure)
			delete this->pLayerStructure;
	}


	//===========================
	// 初期化
	//===========================
	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::Initialize(void)
	{
		for(auto& connectInfo : this->lpConnectInfo)
		{
			connectInfo.pLayerData->Initialize();
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 初期化. 各ニューロンの値をランダムに初期化
		@param	i_config			設定情報
		@oaram	i_inputDataStruct	入力データ構造情報
		@return	成功した場合0 */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct)
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
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize)
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

		// 内部保有のレイヤーデータを削除
		for(auto it : this->lpLayerData)
			delete it.second;
		this->lpLayerData.clear();

		// 初期化する
		this->Initialize();
		
		// レイヤーの数
		U32 layerDataCount = 0;
		memcpy(&layerDataCount, &i_lpBuffer[readBufferByte], sizeof(U32));
		readBufferByte += sizeof(U32);

		// 各レイヤーデータ
		for(U32 layerDataNum=0; layerDataNum<layerDataCount; layerDataNum++)
		{
			// レイヤー種別コード
			Gravisbell::GUID typeCode;
			memcpy(&typeCode, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
			readBufferByte += sizeof(Gravisbell::GUID);

			// レイヤーGUID
			Gravisbell::GUID guid;
			memcpy(&guid, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
			readBufferByte += sizeof(Gravisbell::GUID);

			// レイヤー本体
			auto pLayerDLL = this->layerDLLManager.GetLayerDLLByGUID(typeCode);
			if(pLayerDLL == NULL)
				return ErrorCode::ERROR_CODE_DLL_NOTFOUND;
			S32 useBufferSize = 0;
			auto pLayerData = pLayerDLL->CreateLayerDataFromBuffer(guid, &i_lpBuffer[readBufferByte], readBufferByte, useBufferSize);
			if(pLayerData == NULL)
				return ErrorCode::ERROR_CODE_LAYER_CREATE;
			readBufferByte += useBufferSize;

			// レイヤーデータ一覧に追加
			this->lpLayerData[guid] = pLayerData;
		}

		// レイヤー接続情報
		{
			// レイヤー接続情報数
			U32 connectDataCount = 0;
			memcpy(&connectDataCount, &i_lpBuffer[readBufferByte], sizeof(U32));
			readBufferByte += sizeof(U32);

			// レイヤー接続情報
			for(U32 connectDataNum=0; connectDataNum<connectDataCount; connectDataNum++)
			{
				// レイヤーのGUID
				Gravisbell::GUID layerGUID;
				memcpy(&layerGUID, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
				readBufferByte += sizeof(Gravisbell::GUID);

				// レイヤーデータのGUID
				Gravisbell::GUID layerDataGUID;
				memcpy(&layerDataGUID, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
				readBufferByte += sizeof(Gravisbell::GUID);

				// レイヤーデータ取得
				auto pLayerData = this->lpLayerData[layerDataGUID];
				if(pLayerData == NULL)
					return ErrorCode::ERROR_CODE_LAYER_CREATE;

				// レイヤー接続情報を作成
				LayerConnect layerConnect(layerGUID, pLayerData);

				// 入力レイヤーの数
				U32 inputLayerCount = 0;
				memcpy(&inputLayerCount, &i_lpBuffer[readBufferByte], sizeof(U32));
				readBufferByte += sizeof(U32);

				// 入力レイヤー
				for(U32 inputLayerNum=0; inputLayerNum<inputLayerCount; inputLayerNum++)
				{
					Gravisbell::GUID inputGUID;
					memcpy(&inputGUID, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
					readBufferByte += sizeof(Gravisbell::GUID);

					layerConnect.lpInputLayerGUID.push_back(inputGUID);
				}

				// バイパス入力レイヤーの数
				U32 bypassLayerCount = 0;
				memcpy(&bypassLayerCount, &i_lpBuffer[readBufferByte], sizeof(U32));
				readBufferByte += sizeof(U32);

				// バイパス入力レイヤー
				for(U32 bypassLayerNum=0; bypassLayerNum<bypassLayerCount; bypassLayerNum++)
				{
					Gravisbell::GUID bypassGUID;
					memcpy(&bypassGUID, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
					readBufferByte += sizeof(Gravisbell::GUID);

					layerConnect.lpBypassLayerGUID.push_back(bypassGUID);
				}

				// レイヤーを接続
				this->lpConnectInfo.push_back(layerConnect);
			}
		}

		// 出力レイヤーGUID
		memcpy(&this->outputLayerGUID, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
		readBufferByte += sizeof(Gravisbell::GUID);

		o_useBufferSize = readBufferByte;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// 共通制御
	//===========================
	/** レイヤー固有のGUIDを取得する */
	Gravisbell::GUID FeedforwardNeuralNetwork_LayerData_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** レイヤー種別識別コードを取得する.
		@param o_layerCode	格納先バッファ
		@return 成功した場合0 */
	Gravisbell::GUID FeedforwardNeuralNetwork_LayerData_Base::GetLayerCode(void)const
	{
		Gravisbell::GUID layerCode;
		::GetLayerCode(layerCode);

		return layerCode;
	}

	/** レイヤーDLLマネージャの取得 */
	const ILayerDLLManager& FeedforwardNeuralNetwork_LayerData_Base::GetLayerDLLManager(void)const
	{
		return this->layerDLLManager;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetUseBufferByteCount()const
	{
		U32 bufferSize = 0;

		if(pLayerStructure == NULL)
			return 0;

		// レイヤーデータの一覧を作成する
		std::map<Gravisbell::GUID, ILayerData*> lpTmpLayerData;
		{
			// 本体を保有しているレイヤー
			for(auto& it : this->lpLayerData)
				lpTmpLayerData[it.first] = it.second;
			// 接続レイヤー
			for(auto& it : this->lpConnectInfo)
				lpTmpLayerData[it.pLayerData->GetGUID()] = it.pLayerData;
		}


		// 入力データ構造
		bufferSize += sizeof(this->inputDataStruct);

		// 設定情報
		bufferSize += pLayerStructure->GetUseBufferByteCount();

		// レイヤーの数
		bufferSize += sizeof(U32);

		// 各レイヤーデータ
		for(auto& it : lpTmpLayerData)
		{
			// レイヤー種別コード
			bufferSize += sizeof(Gravisbell::GUID);

			// レイヤーGUID
			bufferSize += sizeof(Gravisbell::GUID);

			// レイヤー本体
			bufferSize += it.second->GetUseBufferByteCount();
		}

		// レイヤー接続情報
		{
			// レイヤー接続情報数
			bufferSize += sizeof(U32);

			// レイヤー接続情報
			for(auto& it : this->lpConnectInfo)
			{
				// レイヤーのGUID
				bufferSize += sizeof(Gravisbell::GUID);

				// レイヤーデータのGUID
				bufferSize += sizeof(Gravisbell::GUID);

				// 入力レイヤーの数
				bufferSize += sizeof(U32);

				// 入力レイヤー
				bufferSize += sizeof(Gravisbell::GUID) * (U32)it.lpInputLayerGUID.size();

				// バイパス入力レイヤーの数
				bufferSize += sizeof(U32);

				// バイパス入力レイヤー
				bufferSize += sizeof(Gravisbell::GUID) * (U32)it.lpBypassLayerGUID.size();
			}
		}

		// 出力レイヤーGUID
		bufferSize += sizeof(Gravisbell::GUID);


		return bufferSize;
	}

	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S32 FeedforwardNeuralNetwork_LayerData_Base::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		// レイヤーデータの一覧を作成する
		std::map<Gravisbell::GUID, ILayerData*> lpTmpLayerData;
		{
			// 本体を保有しているレイヤー
			for(auto& it : this->lpLayerData)
				lpTmpLayerData[it.first] = it.second;
			// 接続レイヤー
			for(auto& it : this->lpConnectInfo)
				lpTmpLayerData[it.pLayerData->GetGUID()] = it.pLayerData;
		}


		int writeBufferByte = 0;

		U32 tmpCount = 0;
		Gravisbell::GUID tmpGUID;

		// 入力データ構造
		memcpy(&o_lpBuffer[writeBufferByte], &this->inputDataStruct, sizeof(this->inputDataStruct));
		writeBufferByte += sizeof(this->inputDataStruct);

		// 設定情報
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		// レイヤーの数
		tmpCount = (U32)lpTmpLayerData.size();
		memcpy(&o_lpBuffer[writeBufferByte], &tmpCount, sizeof(U32));
		writeBufferByte += sizeof(U32);

		// 各レイヤーデータ
		for(auto& it : lpTmpLayerData)
		{
			// レイヤー種別コード
			tmpGUID = it.second->GetLayerCode();
			memcpy(&o_lpBuffer[writeBufferByte], &tmpGUID, sizeof(Gravisbell::GUID));
			writeBufferByte += sizeof(Gravisbell::GUID);

			// レイヤーGUID
			tmpGUID = it.second->GetGUID();
			memcpy(&o_lpBuffer[writeBufferByte], &tmpGUID, sizeof(Gravisbell::GUID));
			writeBufferByte += sizeof(Gravisbell::GUID);

			// テスト用
#ifdef _DEBUG
			U32 useBufferByte  = it.second->GetUseBufferByteCount();
			U32 useBufferByte2 = it.second->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
			if(useBufferByte != useBufferByte2)
			{
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
			}
#endif

			// レイヤー本体
			writeBufferByte += it.second->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
		}

		// レイヤー接続情報
		{
			// レイヤー接続情報数
			tmpCount = (U32)this->lpConnectInfo.size();
			memcpy(&o_lpBuffer[writeBufferByte], &tmpCount, sizeof(U32));
			writeBufferByte += sizeof(U32);

			// レイヤー接続情報
			for(auto& it : this->lpConnectInfo)
			{
				// レイヤーのGUID
				tmpGUID = it.guid;
				memcpy(&o_lpBuffer[writeBufferByte], &tmpGUID, sizeof(Gravisbell::GUID));
				writeBufferByte += sizeof(Gravisbell::GUID);

				// レイヤーデータのGUID
				tmpGUID = it.pLayerData->GetGUID();
				memcpy(&o_lpBuffer[writeBufferByte], &tmpGUID, sizeof(Gravisbell::GUID));
				writeBufferByte += sizeof(Gravisbell::GUID);

				// 入力レイヤーの数
				tmpCount = (U32)it.lpInputLayerGUID.size();
				memcpy(&o_lpBuffer[writeBufferByte], &tmpCount, sizeof(U32));
				writeBufferByte += sizeof(U32);

				// 入力レイヤー
				for(auto guid : it.lpInputLayerGUID)
				{
					memcpy(&o_lpBuffer[writeBufferByte], &guid, sizeof(Gravisbell::GUID));
					writeBufferByte += sizeof(Gravisbell::GUID);
				}

				// バイパス入力レイヤーの数
				tmpCount = (U32)it.lpBypassLayerGUID.size();
				memcpy(&o_lpBuffer[writeBufferByte], &tmpCount, sizeof(U32));
				writeBufferByte += sizeof(U32);

				// バイパス入力レイヤー
				for(auto guid : it.lpBypassLayerGUID)
				{
					memcpy(&o_lpBuffer[writeBufferByte], &guid, sizeof(Gravisbell::GUID));
					writeBufferByte += sizeof(Gravisbell::GUID);
				}
			}
		}

		// 出力レイヤーGUID
		tmpGUID = this->outputLayerGUID;
		memcpy(&o_lpBuffer[writeBufferByte], &tmpGUID, sizeof(Gravisbell::GUID));
		writeBufferByte += sizeof(Gravisbell::GUID);


		return writeBufferByte;
	}


	//===========================
	// レイヤー設定
	//===========================
	/** 設定情報を設定 */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::SetLayerConfig(const SettingData::Standard::IData& config)
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


		return ERROR_CODE_NONE;
	}
	/** レイヤーの設定情報を取得する */
	const SettingData::Standard::IData* FeedforwardNeuralNetwork_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}


	//===========================
	// 入力レイヤー関連
	//===========================
	/** 入力データ構造を取得する.
		@return	入力データ構造 */
	IODataStruct FeedforwardNeuralNetwork_LayerData_Base::GetInputDataStruct()const
	{
		return this->inputDataStruct;
	}

	/** 入力バッファ数を取得する. */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetInputBufferCount()const
	{
		return this->inputDataStruct.GetDataCount();
	}


	//===========================
	// 出力レイヤー関連
	//===========================
	/** 出力データ構造を取得する */
	IODataStruct FeedforwardNeuralNetwork_LayerData_Base::GetOutputDataStruct()const
	{
		auto pConnectLayer = this->GetLayerByGUID(this->outputLayerGUID);
		if(pConnectLayer == NULL)
			return IODataStruct();

		if(const ISingleOutputLayerData* pLayerData = dynamic_cast<const ISingleOutputLayerData*>(pConnectLayer->pLayerData))
		{
			return pLayerData->GetOutputDataStruct();
		}

		return IODataStruct();
	}

	/** 出力バッファ数を取得する */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetOutputBufferCount()const
	{
		return this->GetOutputDataStruct().GetDataCount();
	}



	//===========================
	// レイヤー作成
	//===========================
	/** 作成された新規ニューラルネットワークに対して内部レイヤーを追加する */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddConnectionLayersToNeuralNetwork(class FeedforwardNeuralNetwork_Base& neuralNetwork)
	{
		ErrorCode err;

		// 全レイヤーを追加する
		for(auto it : this->lpConnectInfo)
		{
			err = neuralNetwork.AddLayer(it.pLayerData->CreateLayer(it.guid));
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}

		// レイヤー間の接続を設定する
		for(auto it_connectInfo : this->lpConnectInfo)
		{
			// 入力レイヤー
			for(auto inputGUID : it_connectInfo.lpInputLayerGUID)
			{
				err = neuralNetwork.AddInputLayerToLayer(it_connectInfo.guid, inputGUID);
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;
			}

			// バイパスレイヤー
			for(auto bypassGUID : it_connectInfo.lpBypassLayerGUID)
			{
				err = neuralNetwork.AddBypassLayerToLayer(it_connectInfo.guid, bypassGUID);
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;
			}
		}

		// 出力レイヤーを接続する
		err = neuralNetwork.SetOutputLayerGUID(this->outputLayerGUID);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// レイヤー間の接続状態を確認
		Gravisbell::GUID errorLayerGUID;
		return neuralNetwork.CheckAllConnection(errorLayerGUID);
	}


	//====================================
	// レイヤーの追加/削除/管理
	//====================================
	/** レイヤーデータを追加する.
		@param	i_guid			追加するレイヤーに割り当てられるGUID.
		@param	i_pLayerData	追加するレイヤーデータのアドレス. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddLayer(const Gravisbell::GUID& i_guid, ILayerData* i_pLayerData)
	{
		// レイヤーを検索
		if(this->GetLayerByGUID(i_guid))
			return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;

		// 追加
		this->lpConnectInfo.push_back(LayerConnect(i_guid, i_pLayerData));

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** レイヤーデータを削除する.
		@param i_guid	削除するレイヤーのGUID */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::EraseLayer(const Gravisbell::GUID& i_guid)
	{
		// 削除レイヤーを検索
		auto it = this->lpConnectInfo.begin();
		while(it != this->lpConnectInfo.end())
		{
			if(it->guid == i_guid)
				break;
		}
		if(it == this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;

		// 削除対象のレイヤーを入力に持つ場合は削除
		for(auto& it_search : this->lpConnectInfo)
		{
			this->EraseInputLayerFromLayer(it_search.guid, i_guid);
			this->EraseBypassLayerFromLayer(it_search.guid, i_guid);
		}

		// レイヤーデータ本体を持つ場合は削除
		{
			auto it_pLayerData = this->lpLayerData.find(it->pLayerData->GetGUID());
			if(it_pLayerData != this->lpLayerData.end())
			{
				// 削除対象のレイヤー以外に同一のレイヤーデータを使用しているレイヤーが存在しないか確認
				bool onCanErase = true;
				for(auto& it_search : this->lpConnectInfo)
				{
					if(it_search.guid != i_guid)
					{
						if(it_search.pLayerData->GetGUID() == it_pLayerData->first)
						{
							onCanErase = false;
							break;
						}
					}
				}

				// レイヤーデータを削除
				if(onCanErase)
				{
					delete it_pLayerData->second;
					this->lpLayerData.erase(it_pLayerData);
				}
			}
		}

		// 接続を削除
		this->lpConnectInfo.erase(it);

		// 出力対象レイヤーの場合解除する
		if(this->outputLayerGUID == i_guid)
			this->outputLayerGUID = Gravisbell::GUID();

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** レイヤーデータを全削除する */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::EraseAllLayer()
	{
		// 接続情報を全削除
		this->lpConnectInfo.clear();

		// レイヤーデータ本体を削除
		for(auto it : this->lpLayerData)
		{
			delete it.second;
		}
		this->lpLayerData.clear();

		// 出力対象レイヤーの場合解除する
		this->outputLayerGUID = Gravisbell::GUID();

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 登録されているレイヤー数を取得する */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetLayerCount()
	{
		return (U32)this->lpConnectInfo.size();
	}
	/** レイヤーのGUIDを番号指定で取得する */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::GetLayerGUIDbyNum(U32 i_layerNum, Gravisbell::GUID& o_guid)
	{
		if(i_layerNum >= this->lpConnectInfo.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		auto it = this->lpConnectInfo.begin();
		for(U32 i=0; i<i_layerNum; i++)
			it++;

		o_guid = it->guid;

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 登録されているレイヤーを番号指定で取得する */
	FeedforwardNeuralNetwork_LayerData_Base::LayerConnect* FeedforwardNeuralNetwork_LayerData_Base::GetLayerByNum(U32 i_layerNum)
	{
		if(i_layerNum >= this->lpConnectInfo.size())
			return NULL;

		return &this->lpConnectInfo[i_layerNum];
	}
	/** 登録されているレイヤーをGUID指定で取得する */
	FeedforwardNeuralNetwork_LayerData_Base::LayerConnect* FeedforwardNeuralNetwork_LayerData_Base::GetLayerByGUID(const Gravisbell::GUID& i_guid)
	{
		for(U32 layerNum=0; layerNum<this->lpConnectInfo.size(); layerNum++)
		{
			if(this->lpConnectInfo[layerNum].guid == i_guid)
				return &this->lpConnectInfo[layerNum];
		}
		return NULL;
	}
	FeedforwardNeuralNetwork_LayerData_Base::LayerConnect* FeedforwardNeuralNetwork_LayerData_Base::GetLayerByGUID(const Gravisbell::GUID& i_guid)const
	{
		return (const_cast<FeedforwardNeuralNetwork_LayerData_Base*>(this))->GetLayerByGUID(i_guid);
	}

	/** 登録されているレイヤーデータを番号指定で取得する */
	ILayerData* FeedforwardNeuralNetwork_LayerData_Base::GetLayerDataByNum(U32 i_layerNum)
	{
		Gravisbell::GUID layerGUID;
		ErrorCode err = this->GetLayerGUIDbyNum(i_layerNum, layerGUID);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return NULL;

		return this->GetLayerDataByGUID(layerGUID);
	}
	/** 登録されているレイヤーデータをGUID指定で取得する */
	ILayerData* FeedforwardNeuralNetwork_LayerData_Base::GetLayerDataByGUID(const Gravisbell::GUID& i_guid)
	{
		auto pLayerConnet = this->GetLayerByGUID(i_guid);
		if(pLayerConnet == NULL)
			return NULL;

		return pLayerConnet->pLayerData;
	}


	//====================================
	// 入出力レイヤー
	//====================================
	/** 入力信号に割り当てられているGUIDを取得する */
	GUID FeedforwardNeuralNetwork_LayerData_Base::GetInputGUID()
	{
		return this->inputLayerGUID;
	}

	/** 出力信号レイヤーを設定する */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::SetOutputLayerGUID(const Gravisbell::GUID& i_guid)
	{
		auto pLayerConnet = this->GetLayerByGUID(i_guid);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		if(dynamic_cast<ISingleOutputLayerData*>(pLayerConnet->pLayerData))
		{
			this->outputLayerGUID = i_guid;

			return ErrorCode::ERROR_CODE_NONE;
		}

		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** 出力信号レイヤーのGUIDを取得する */
	Gravisbell::GUID FeedforwardNeuralNetwork_LayerData_Base::GetOutputLayerGUID()
	{
		return this->outputLayerGUID;
	}


	//====================================
	// レイヤーの接続
	//====================================
	/** レイヤーに入力レイヤーを追加する.
		@param	receiveLayer	入力を受け取るレイヤー
		@param	postLayer		入力を渡す(出力する)レイヤー. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddInputLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer)
	{
		// レイヤーの存在を確認
		auto pLayerConnet = this->GetLayerByGUID(receiveLayer);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 同一レイヤーが追加済でないことを確認
		{
			auto it = pLayerConnet->lpInputLayerGUID.begin();
			while(it != pLayerConnet->lpInputLayerGUID.end())
			{
				if(*it == postLayer)
					return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;
				it++;
			}
		}

		// レイヤーを追加
		pLayerConnet->lpInputLayerGUID.push_back(postLayer);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** レイヤーにバイパスレイヤーを追加する.
		@param	receiveLayer	入力を受け取るレイヤー
		@param	postLayer		入力を渡す(出力する)レイヤー. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddBypassLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer)
	{
		// レイヤーの存在を確認
		auto pLayerConnet = this->GetLayerByGUID(receiveLayer);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 同一レイヤーが追加済でないことを確認
		{
			auto it = pLayerConnet->lpInputLayerGUID.begin();
			while(it != pLayerConnet->lpInputLayerGUID.end())
			{
				if(*it == postLayer)
					return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;
				it++;
			}
		}

		// レイヤーを追加
		pLayerConnet->lpBypassLayerGUID.push_back(postLayer);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** レイヤーから入力レイヤーを削除する. 
		@param	receiveLayer	入力を受け取るレイヤー
		@param	postLayer		入力を渡す(出力する)レイヤー. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::EraseInputLayerFromLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer)
	{
		auto pLayerConnect = this->GetLayerByGUID(receiveLayer);
		if(pLayerConnect == NULL)
			return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;

		auto it = pLayerConnect->lpInputLayerGUID.begin();
		while(it != pLayerConnect->lpInputLayerGUID.end())
		{
			if(*it == postLayer)
			{
				it = pLayerConnect->lpInputLayerGUID.erase(it);
			}
			else
			{
				it++;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** レイヤーからバイパスレイヤーを削除する.
		@param	receiveLayer	入力を受け取るレイヤー
		@param	postLayer		入力を渡す(出力する)レイヤー. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::EraseBypassLayerFromLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer)
	{
		auto pLayerConnect = this->GetLayerByGUID(receiveLayer);
		if(pLayerConnect == NULL)
			return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;

		auto it = pLayerConnect->lpBypassLayerGUID.begin();
		while(it != pLayerConnect->lpBypassLayerGUID.end())
		{
			if(*it == postLayer)
			{
				it = pLayerConnect->lpBypassLayerGUID.erase(it);
			}
			else
			{
				it++;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** レイヤーの入力レイヤー設定をリセットする.
		@param	layerGUID	リセットするレイヤーのGUID. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::ResetInputLayer(const Gravisbell::GUID& i_layerGUID)
	{
		// レイヤーの存在を確認
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 削除処理
		pLayerConnet->lpInputLayerGUID.clear();

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** レイヤーのバイパスレイヤー設定をリセットする.
		@param	layerGUID	リセットするレイヤーのGUID. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::ResetBypassLayer(const Gravisbell::GUID& i_layerGUID)
	{
		// レイヤーの存在を確認
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 削除処理
		pLayerConnet->lpBypassLayerGUID.clear();

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** レイヤーに接続している入力レイヤーの数を取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID. */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetInputLayerCount(const Gravisbell::GUID& i_layerGUID)
	{
		// レイヤーの存在を確認
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return 0;

		return (U32)pLayerConnet->lpInputLayerGUID.size();
	}
	/** レイヤーに接続しているバイパスレイヤーの数を取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID. */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID)
	{
		// レイヤーの存在を確認
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return 0;

		return (U32)pLayerConnet->lpBypassLayerGUID.size();
	}
	/** レイヤーに接続している出力レイヤーの数を取得する */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetOutputLayerCount(const Gravisbell::GUID& i_layerGUID)
	{
		// レイヤーの存在を確認
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 対象レイヤーを入力レイヤーに持つレイヤー数を数える
		U32 outputCount = 0;
		for(auto it : this->lpConnectInfo)
		{
			for(U32 inputNum=0; inputNum<pLayerConnet->lpInputLayerGUID.size(); inputNum++)
			{
				if(pLayerConnet->lpInputLayerGUID[inputNum] == i_layerGUID)
					outputCount++;
			}
		}

		return outputCount;
	}

	/** レイヤーに接続している入力レイヤーのGUIDを番号指定で取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
		@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::GetInputLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)
	{
		// レイヤーの存在を確認
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 入力レイヤーの数を確認
		if(i_inputNum >= pLayerConnet->lpInputLayerGUID.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		// イテレータ進行
		auto it = pLayerConnet->lpInputLayerGUID.begin();
		for(U32 i=0; i<i_inputNum; i++)
			it++;

		o_postLayerGUID = *it;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
		@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::GetBypassLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)
	{
		// レイヤーの存在を確認
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 入力レイヤーの数を確認
		if(i_inputNum >= pLayerConnet->lpBypassLayerGUID.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		// イテレータ進行
		auto it = pLayerConnet->lpBypassLayerGUID.begin();
		for(U32 i=0; i<i_inputNum; i++)
			it++;

		o_postLayerGUID = *it;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** レイヤーに接続している出力レイヤーのGUIDを番号指定で取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID.
		@param	i_outputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
		@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::GetOutputLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_outputNum, Gravisbell::GUID& o_postLayerGUID)
	{
		// レイヤーの存在を確認
		auto pLayerConnet = this->GetLayerByGUID(i_layerGUID);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 対象レイヤーを入力レイヤーに持つレイヤー数を数えて番号が一致したら終了
		U32 outputNum = 0;
		for(auto it : this->lpConnectInfo)
		{
			for(U32 inputNum=0; inputNum<pLayerConnet->lpInputLayerGUID.size(); inputNum++)
			{
				if(pLayerConnet->lpInputLayerGUID[inputNum] == i_layerGUID)
				{
					if(outputNum == i_outputNum)
					{
						o_postLayerGUID = pLayerConnet->guid;
						return ErrorCode::ERROR_CODE_NONE;
					}
					outputNum++;
				}
			}
		}

		return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
	}


}	// NeuralNetwork
}	// Layer
}	// Gravisbell
