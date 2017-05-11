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
			connectInfo.second.pLayerData->Initialize();
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

					layerConnect.lpInputLayerGUID.insert(inputGUID);
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

					layerConnect.lpBypassLayerGUID.insert(bypassGUID);
				}

				// レイヤーを接続
				this->lpConnectInfo[layerGUID] = layerConnect;
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
		std::map<Gravisbell::GUID, INNLayerData*> lpTmpLayerData;
		{
			// 本体を保有しているレイヤー
			for(auto& it : this->lpLayerData)
				lpTmpLayerData[it.first] = it.second;
			// 接続レイヤー
			for(auto& it : this->lpConnectInfo)
				lpTmpLayerData[it.second.pLayerData->GetGUID()] = it.second.pLayerData;
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
				bufferSize += sizeof(Gravisbell::GUID) * (U32)it.second.lpInputLayerGUID.size();

				// バイパス入力レイヤーの数
				bufferSize += sizeof(U32);

				// バイパス入力レイヤー
				bufferSize += sizeof(Gravisbell::GUID) * (U32)it.second.lpBypassLayerGUID.size();
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
		std::map<Gravisbell::GUID, INNLayerData*> lpTmpLayerData;
		{
			// 本体を保有しているレイヤー
			for(auto& it : this->lpLayerData)
				lpTmpLayerData[it.first] = it.second;
			// 接続レイヤー
			for(auto& it : this->lpConnectInfo)
				lpTmpLayerData[it.second.pLayerData->GetGUID()] = it.second.pLayerData;
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
				tmpGUID = it.first;
				memcpy(&o_lpBuffer[writeBufferByte], &tmpGUID, sizeof(Gravisbell::GUID));
				writeBufferByte += sizeof(Gravisbell::GUID);

				// レイヤーデータのGUID
				tmpGUID = it.second.pLayerData->GetGUID();
				memcpy(&o_lpBuffer[writeBufferByte], &tmpGUID, sizeof(Gravisbell::GUID));
				writeBufferByte += sizeof(Gravisbell::GUID);

				// 入力レイヤーの数
				tmpCount = (U32)it.second.lpInputLayerGUID.size();
				memcpy(&o_lpBuffer[writeBufferByte], &tmpCount, sizeof(U32));
				writeBufferByte += sizeof(U32);

				// 入力レイヤー
				for(auto guid : it.second.lpInputLayerGUID)
				{
					memcpy(&o_lpBuffer[writeBufferByte], &guid, sizeof(Gravisbell::GUID));
					writeBufferByte += sizeof(Gravisbell::GUID);
				}

				// バイパス入力レイヤーの数
				tmpCount = (U32)it.second.lpBypassLayerGUID.size();
				memcpy(&o_lpBuffer[writeBufferByte], &tmpCount, sizeof(U32));
				writeBufferByte += sizeof(U32);

				// バイパス入力レイヤー
				for(auto guid : it.second.lpBypassLayerGUID)
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
		auto it = this->lpConnectInfo.find(this->outputLayerGUID);
		if(it == this->lpConnectInfo.end())
			return IODataStruct();

		return it->second.pLayerData->GetOutputDataStruct();
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
			err = neuralNetwork.AddLayer(it.second.pLayerData->CreateLayer(it.first));
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}

		// レイヤー間の接続を設定する
		for(auto it_connectInfo : this->lpConnectInfo)
		{
			// 入力レイヤー
			for(auto inputGUID : it_connectInfo.second.lpInputLayerGUID)
			{
				err = neuralNetwork.AddInputLayerToLayer(it_connectInfo.first, inputGUID);
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;
			}

			// バイパスレイヤー
			for(auto bypassGUID : it_connectInfo.second.lpBypassLayerGUID)
			{
				err = neuralNetwork.AddBypassLayerToLayer(it_connectInfo.first, bypassGUID);
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
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddLayer(const Gravisbell::GUID& i_guid, INNLayerData* i_pLayerData)
	{
		// レイヤーを検索
		if(this->lpConnectInfo.count(i_guid) != 0)
			return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;

		// 追加
		this->lpConnectInfo[i_guid] = LayerConnect(i_guid, i_pLayerData);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** レイヤーデータを削除する.
		@param i_guid	削除するレイヤーのGUID */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::EraseLayer(const Gravisbell::GUID& i_guid)
	{
		// 削除レイヤーを検索
		auto it = this->lpConnectInfo.find(i_guid);
		if(it == this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;

		// 削除対象のレイヤーを入力に持つ場合は削除
		for(auto& it_search : this->lpConnectInfo)
		{
			it_search.second.lpInputLayerGUID.erase(i_guid);
			it_search.second.lpBypassLayerGUID.erase(i_guid);
		}

		// レイヤーデータ本体を持つ場合は削除
		{
			auto it_pLayerData = this->lpLayerData.find(it->second.pLayerData->GetGUID());
			if(it_pLayerData != this->lpLayerData.end())
			{
				// 削除対象のレイヤー以外に同一のレイヤーデータを使用しているレイヤーが存在しないか確認
				bool onCanErase = true;
				for(auto& it_search : this->lpConnectInfo)
				{
					if(it_search.first != i_guid)
					{
						if(it_search.second.pLayerData->GetGUID() == it_pLayerData->first)
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

		o_guid = it->first;

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 登録されているレイヤーデータを番号指定で取得する */
	INNLayerData* FeedforwardNeuralNetwork_LayerData_Base::GetLayerDataByNum(U32 i_layerNum)
	{
		Gravisbell::GUID layerGUID;
		ErrorCode err = this->GetLayerGUIDbyNum(i_layerNum, layerGUID);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return NULL;

		return this->GetLayerDataByGUID(layerGUID);
	}
	/** 登録されているレイヤーデータをGUID指定で取得する */
	INNLayerData* FeedforwardNeuralNetwork_LayerData_Base::GetLayerDataByGUID(const Gravisbell::GUID& i_guid)
	{
		auto it = this->lpConnectInfo.find(i_guid);
		if(it == this->lpConnectInfo.end())
			return NULL;

		return it->second.pLayerData;
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
		if(this->lpConnectInfo.count(i_guid) == 0)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		this->outputLayerGUID = i_guid;

		return ErrorCode::ERROR_CODE_NONE;
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
		auto it_receive = this->lpConnectInfo.find(receiveLayer);
		if(it_receive == this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 同一レイヤーが追加済でないことを確認
		if(it_receive->second.lpInputLayerGUID.count(postLayer) != 0)
			return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;

		// レイヤーを追加
		it_receive->second.lpInputLayerGUID.insert(postLayer);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** レイヤーにバイパスレイヤーを追加する.
		@param	receiveLayer	入力を受け取るレイヤー
		@param	postLayer		入力を渡す(出力する)レイヤー. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddBypassLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer)
	{
		// レイヤーの存在を確認
		auto it_receive = this->lpConnectInfo.find(receiveLayer);
		if(it_receive == this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 同一レイヤーが追加済でないことを確認
		if(it_receive->second.lpBypassLayerGUID.count(postLayer) != 0)
			return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;

		// レイヤーを追加
		it_receive->second.lpBypassLayerGUID.insert(postLayer);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** レイヤーの入力レイヤー設定をリセットする.
		@param	layerGUID	リセットするレイヤーのGUID. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::ResetInputLayer(const Gravisbell::GUID& i_layerGUID)
	{
		// レイヤーの存在を確認
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer == this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 削除処理
		it_layer->second.lpInputLayerGUID.clear();

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** レイヤーのバイパスレイヤー設定をリセットする.
		@param	layerGUID	リセットするレイヤーのGUID. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::ResetBypassLayer(const Gravisbell::GUID& i_layerGUID)
	{
		// レイヤーの存在を確認
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer == this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 削除処理
		it_layer->second.lpBypassLayerGUID.clear();

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** レイヤーに接続している入力レイヤーの数を取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID. */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetInputLayerCount(const Gravisbell::GUID& i_layerGUID)
	{
		// レイヤーの存在を確認
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer == this->lpConnectInfo.end())
			return 0;

		return (U32)it_layer->second.lpInputLayerGUID.size();
	}
	/** レイヤーに接続しているバイパスレイヤーの数を取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID. */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID)
	{
		// レイヤーの存在を確認
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer == this->lpConnectInfo.end())
			return 0;

		return (U32)it_layer->second.lpBypassLayerGUID.size();
	}
	/** レイヤーに接続している出力レイヤーの数を取得する */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetOutputLayerCount(const Gravisbell::GUID& i_layerGUID)
	{
		// レイヤーの存在を確認
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer == this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 対象レイヤーを入力レイヤーに持つレイヤー数を数える
		U32 outputCount = 0;
		for(auto it : this->lpConnectInfo)
		{
			if(it.second.lpInputLayerGUID.count(i_layerGUID) != 0)
				outputCount++;
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
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer == this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 入力レイヤーの数を確認
		if(i_inputNum >= it_layer->second.lpInputLayerGUID.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		// イテレータ進行
		auto it = it_layer->second.lpInputLayerGUID.begin();
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
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer == this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 入力レイヤーの数を確認
		if(i_inputNum >= it_layer->second.lpBypassLayerGUID.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		// イテレータ進行
		auto it = it_layer->second.lpBypassLayerGUID.begin();
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
		auto it_layer = this->lpConnectInfo.find(i_layerGUID);
		if(it_layer == this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 対象レイヤーを入力レイヤーに持つレイヤー数を数えて番号が一致したら終了
		U32 outputNum = 0;
		for(auto it : this->lpConnectInfo)
		{
			if(it.second.lpInputLayerGUID.count(i_layerGUID) != 0)
			{
				if(outputNum == i_outputNum)
				{
					o_postLayerGUID = it.first;
					return ErrorCode::ERROR_CODE_NONE;
				}
				outputNum++;
			}
		}

		return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
	}


}	// NeuralNetwork
}	// Layer
}	// Gravisbell
