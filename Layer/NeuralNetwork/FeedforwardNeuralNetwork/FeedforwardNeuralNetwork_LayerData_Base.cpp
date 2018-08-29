//======================================
// フィードフォワードニューラルネットワークの処理レイヤーのデータ
// 複数のレイヤーを内包し、処理する
//======================================
#include"stdafx.h"

#include"FeedforwardNeuralNetwork_LayerData_Base.h"
#include"FeedforwardNeuralNetwork_FUNC.hpp"
#include"FeedforwardNeuralNetwork_Base.h"

#include<boost/uuid/uuid_generators.hpp>

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
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::Initialize(const SettingData::Standard::IData& i_data)
	{
		ErrorCode err;

		// 設定情報の登録
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// 内部保有のレイヤーデータを削除
		for(auto it : this->lpLayerData)
			delete it.second;
		this->lpLayerData.clear();

		// 入力レイヤー一覧を作成
		this->lpInputLayerGUID.resize(this->layerStructure.inputLayerCount);
		for(size_t i=0; i<this->lpInputLayerGUID.size(); i++)
		{
			this->lpInputLayerGUID[i] = Gravisbell::GUID(boost::uuids::random_generator()().data);
		}

		return this->Initialize();
	}
	/** 初期化. バッファからデータを読み込む
		@param i_lpBuffer	読み込みバッファの先頭アドレス.
		@param i_bufferSize	読み込み可能バッファのサイズ.
		@return	成功した場合0 */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize, S64& o_useBufferSize)
	{
		S64 readBufferByte = 0;

		// 設定情報
		S64 useBufferByte = 0;
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(&i_lpBuffer[readBufferByte], i_bufferSize, useBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		readBufferByte += useBufferByte;

		// 初期化
		this->Initialize(*pLayerStructure);

		// 読み込んだバッファを開放
		this->SetLayerConfig(*pLayerStructure);
		delete pLayerStructure;

		// 入力レイヤーGUID
		for(size_t i=0; i<this->lpInputLayerGUID.size(); i++)
		{
			Gravisbell::GUID guid;
			memcpy(&guid, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
			readBufferByte += sizeof(Gravisbell::GUID);

			this->lpInputLayerGUID[i] = guid;
		}

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
			S64 useBufferSize = 0;
			auto pLayerData = pLayerDLL->CreateLayerDataFromBuffer(guid, &i_lpBuffer[readBufferByte], i_bufferSize - readBufferByte, useBufferSize);
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

				// レイヤーの固定フラグ
				bool onFixFlag = false;
				memcpy(&onFixFlag, &i_lpBuffer[readBufferByte], sizeof(onFixFlag));
				readBufferByte += sizeof(onFixFlag);

				// レイヤーデータのGUID
				Gravisbell::GUID layerDataGUID;
				memcpy(&layerDataGUID, &i_lpBuffer[readBufferByte], sizeof(Gravisbell::GUID));
				readBufferByte += sizeof(Gravisbell::GUID);

				// レイヤーデータ取得
				auto pLayerData = this->lpLayerData[layerDataGUID];
				if(pLayerData == NULL)
					return ErrorCode::ERROR_CODE_LAYER_CREATE;

				// レイヤー接続情報を作成
				LayerConnect layerConnect(layerGUID, pLayerData, onFixFlag);

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
				this->lpConnectInfo[layerConnect.guid] = layerConnect;
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
	U64 FeedforwardNeuralNetwork_LayerData_Base::GetUseBufferByteCount()const
	{
		U64 bufferSize = 0;

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
				lpTmpLayerData[it.second.pLayerData->GetGUID()] = it.second.pLayerData;
		}


		// 設定情報
		bufferSize += pLayerStructure->GetUseBufferByteCount();

		// 入力レイヤーGUID一覧
		bufferSize += this->layerStructure.inputLayerCount * sizeof(Gravisbell::GUID);

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

				// レイヤーの固定フラグ
				bufferSize += sizeof(bool);

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
	S64 FeedforwardNeuralNetwork_LayerData_Base::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return -1;

		// レイヤーデータの一覧を作成する
		std::map<Gravisbell::GUID, ILayerData*> lpTmpLayerData;
		{
			// 本体を保有しているレイヤー
			for(auto& it : this->lpLayerData)
				lpTmpLayerData[it.first] = it.second;
			// 接続レイヤー
			for(auto& it : this->lpConnectInfo)
				lpTmpLayerData[it.second.pLayerData->GetGUID()] = it.second.pLayerData;
		}


		S64 writeBufferByte = 0;

		U32 tmpCount = 0;
		Gravisbell::GUID tmpGUID;

		// 設定情報
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		// 入力レイヤーGUID
		for(auto guid : this->lpInputLayerGUID)
		{
			memcpy(&o_lpBuffer[writeBufferByte], &guid, sizeof(Gravisbell::GUID));
			writeBufferByte += sizeof(Gravisbell::GUID);
		}

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
			U64 useBufferByte  = it.second->GetUseBufferByteCount();
			S64 useBufferByte2 = it.second->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
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
				tmpGUID = it.second.guid;
				memcpy(&o_lpBuffer[writeBufferByte], &tmpGUID, sizeof(Gravisbell::GUID));
				writeBufferByte += sizeof(Gravisbell::GUID);

				// レイヤーの固定フラグ
				memcpy(&o_lpBuffer[writeBufferByte], &it.second.onFixFlag, sizeof(it.second.onFixFlag));
				writeBufferByte += sizeof(it.second.onFixFlag);

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

		// 構造体に読み込む
		this->pLayerStructure->WriteToStruct((BYTE*)&this->layerStructure);

		return ERROR_CODE_NONE;
	}
	/** レイヤーの設定情報を取得する */
	const SettingData::Standard::IData* FeedforwardNeuralNetwork_LayerData_Base::GetLayerStructure()const
	{
		return this->pLayerStructure;
	}

	//===========================
	// レイヤー構造
	//===========================
	/** 入力データ構造が使用可能か確認する.
		@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
		@return	使用可能な入力データ構造の場合trueが返る. */
	bool FeedforwardNeuralNetwork_LayerData_Base::CheckCanUseInputDataStruct(const Gravisbell::GUID& i_guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		return this->GetOutputDataStruct(i_guid, i_lpInputDataStruct, i_inputLayerCount).GetDataCount() != 0;
	}
	bool FeedforwardNeuralNetwork_LayerData_Base::CheckCanUseInputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		return this->GetOutputDataStruct(i_lpInputDataStruct, i_inputLayerCount).GetDataCount() != 0;
	}


	/** 出力データ構造を取得する.
		@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
		@return	入力データ構造が不正な場合(x=0,y=0,z=0,ch=0)が返る. */
	IODataStruct FeedforwardNeuralNetwork_LayerData_Base::GetOutputDataStruct(const Gravisbell::GUID& i_guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		// 入力レイヤー数が異なる場合は終了
		if(i_inputLayerCount != this->layerStructure.inputLayerCount)
			return IODataStruct(0,0,0,0);

		this->tmp_lpInputDataStruct = i_lpInputDataStruct;
		this->tmp_inputLayerCount = i_inputLayerCount;
		this->tmp_lpOutputDataStruct.clear();

		return this->GetOutputDataStruct(i_guid);
	}
	/** 出力データ構造を取得する.
		@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
		@return	入力データ構造が不正な場合(x=0,y=0,z=0,ch=0)が返る. */
	IODataStruct FeedforwardNeuralNetwork_LayerData_Base::GetOutputDataStruct(const Gravisbell::GUID& i_guid)
	{
		// 既に計算済みのレイヤーか確認する
		{
			auto it_layer = this->tmp_lpOutputDataStruct.find(i_guid);
			if(it_layer != this->tmp_lpOutputDataStruct.end())
				return it_layer->second;
		}

		for(size_t inputNum=0; inputNum<this->lpInputLayerGUID.size(); inputNum++)
		{
			if(i_guid == this->lpInputLayerGUID[inputNum])
			{
				return this->tmp_lpOutputDataStruct[i_guid] = this->tmp_lpInputDataStruct[inputNum];
			}
		}

		auto pConnectLayer = this->GetLayerByGUID(i_guid);
		if(pConnectLayer == NULL)
			return this->tmp_lpOutputDataStruct[i_guid] = IODataStruct(0,0,0,0);

		// 入力レイヤー
		std::vector<IODataStruct> lpInputDataStruct(pConnectLayer->lpInputLayerGUID.size());
		for(U32 inputLayerNum=0; inputLayerNum<lpInputDataStruct.size(); inputLayerNum++)
		{
			lpInputDataStruct[inputLayerNum] = this->GetOutputDataStruct(pConnectLayer->lpInputLayerGUID[inputLayerNum]);
			if(lpInputDataStruct[inputLayerNum].GetDataCount() == 0)
				return this->tmp_lpOutputDataStruct[i_guid] = IODataStruct(0,0,0,0);
		}
		// バイパスレイヤー
		std::vector<IODataStruct> lpBypassDataStruct(pConnectLayer->lpBypassLayerGUID.size());
		for(U32 inputLayerNum=0; inputLayerNum<lpBypassDataStruct.size(); inputLayerNum++)
		{
			lpBypassDataStruct[inputLayerNum] = this->GetOutputDataStruct(pConnectLayer->lpBypassLayerGUID[inputLayerNum]);
			if(lpBypassDataStruct[inputLayerNum].GetDataCount() == 0)
				return this->tmp_lpOutputDataStruct[i_guid] = IODataStruct(0,0,0,0);
		}

		if(lpInputDataStruct.size() == 0)
			return this->tmp_lpOutputDataStruct[i_guid] = IODataStruct(0,0,0,0);

		// 入力が複数ある場合
		if(lpInputDataStruct.size() > 1)
		{
			// 複数入力を受け付けているかチェック
			if(pConnectLayer->pLayerData->CheckCanUseInputDataStruct(&lpInputDataStruct[0], (U32)lpInputDataStruct.size()))
			{
				return this->tmp_lpOutputDataStruct[i_guid] = pConnectLayer->pLayerData->GetOutputDataStruct(&lpInputDataStruct[0], (U32)lpInputDataStruct.size());
			}
			else
			{
				// CH以外が一致していることを確認
				IODataStruct inputDataStruct = lpInputDataStruct[0];
				for(U32 layerNum=1; layerNum<lpInputDataStruct.size(); layerNum++)
				{
					if(inputDataStruct.x != lpInputDataStruct[layerNum].x)	return this->tmp_lpOutputDataStruct[i_guid] = IODataStruct(0,0,0,0);
					if(inputDataStruct.y != lpInputDataStruct[layerNum].y)	return this->tmp_lpOutputDataStruct[i_guid] = IODataStruct(0,0,0,0);
					if(inputDataStruct.z != lpInputDataStruct[layerNum].z)	return this->tmp_lpOutputDataStruct[i_guid] = IODataStruct(0,0,0,0);

					inputDataStruct.ch += lpInputDataStruct[layerNum].ch;
				}

				return this->tmp_lpOutputDataStruct[i_guid] = pConnectLayer->pLayerData->GetOutputDataStruct(&inputDataStruct, 1);
			}
		}

		return this->tmp_lpOutputDataStruct[i_guid] = pConnectLayer->pLayerData->GetOutputDataStruct(&lpInputDataStruct[0], 1);
	}

	/** 出力データ構造を取得する.
		@param	i_lpInputDataStruct	入力データ構造の配列. GetInputFromLayerCount()の戻り値以上の要素数が必要
		@return	入力データ構造が不正な場合(x=0,y=0,z=0,ch=0)が返る. */
	IODataStruct FeedforwardNeuralNetwork_LayerData_Base::GetOutputDataStruct(const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		if(i_inputLayerCount == 0)
			return false;
		if(i_inputLayerCount > 1)
			return false;
		if(i_lpInputDataStruct == NULL)
			return false;

		return this->GetOutputDataStruct(this->outputLayerGUID, i_lpInputDataStruct, i_inputLayerCount);
	}

	/** 複数出力が可能かを確認する */
	bool FeedforwardNeuralNetwork_LayerData_Base::CheckCanHaveMultOutputLayer(void)
	{
		return false;
	}


	//===========================
	// レイヤー作成
	//===========================
	/** 作成された新規ニューラルネットワークに対して内部レイヤーを追加する */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddConnectionLayersToNeuralNetwork(class FeedforwardNeuralNetwork_Base& neuralNetwork, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
	{
		ErrorCode err;

		// 出力分割レイヤーの識別ID
		static const Gravisbell::GUID SEPARATE_LAYER_GUID(0xc13c30da, 0x056e, 0x46d0, 0x90, 0xfc, 0x60, 0x87, 0x66, 0xfb, 0x43, 0x2e);

		std::map<Gravisbell::GUID, Gravisbell::GUID>	lpSubstitutionLayer;	// 代替レイヤー<元レイヤーGUID, 代替レイヤーGUID>

		// 全レイヤーを追加する
		for(auto it : this->lpConnectInfo)
		{
			// 対象レイヤーに対する入力データ構造一覧を作成
			std::vector<IODataStruct> lpInputDataStruct;
			for(auto inputGUID : it.second.lpInputLayerGUID)
			{
				lpInputDataStruct.push_back(this->GetOutputDataStruct(inputGUID, i_lpInputDataStruct, i_inputLayerCount));
			}
			
			err = neuralNetwork.AddLayer(it.second.pLayerData->CreateLayer(it.second.guid, &lpInputDataStruct[0], (U32)lpInputDataStruct.size(), neuralNetwork.GetTemporaryMemoryManager()), it.second.onFixFlag);

			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}

		// 複数出力を持つレイヤーで、単一出力の能力しか持たないものを探し、代替レイヤーを作成する
		for(auto it : this->lpConnectInfo)
		{
			if(this->GetOutputLayerCount(it.second.guid) > 1)
			{
				ILayerData* pLayerData = it.second.pLayerData;
				if(!pLayerData->CheckCanHaveMultOutputLayer())
				{
					// 単一出力機能しか持たないのに複数出力を持っている

					// 出力分割レイヤ－のDLLを取得
					auto pDLL = this->layerDLLManager.GetLayerDLLByGUID(SEPARATE_LAYER_GUID);
					if(pDLL == NULL)
						return ErrorCode::ERROR_CODE_DLL_NOTFOUND;

					// レイヤー構造情報を作成
					auto pLayerStructure = pDLL->CreateLayerStructureSetting();
					if(pLayerStructure == NULL)
						return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
					auto pItem = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Int*>(pLayerStructure->GetItemByID(L"separateCount"));
					if(pItem == NULL)
						return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;

					pItem->SetValue(this->GetOutputLayerCount(it.second.guid));

					// レイヤーデータを作成
					auto pSubstitutionLayerData = pDLL->CreateLayerData(*pLayerStructure);
					delete pLayerStructure;

					// レイヤーを追加
					ILayerBase* pSubstitutionLayer = NULL;
					err = neuralNetwork.AddTemporaryLayer(pSubstitutionLayerData, &pSubstitutionLayer, &this->GetOutputDataStruct(it.second.guid, i_lpInputDataStruct, i_inputLayerCount), 1, true);
					if(err != ErrorCode::ERROR_CODE_NONE)
						return err;

					lpSubstitutionLayer[it.second.guid] = pSubstitutionLayer->GetGUID();

					// 代替レイヤーの入力を代替先レイヤーに設定
					err = neuralNetwork.AddInputLayerToLayer(pSubstitutionLayer->GetGUID(), it.second.guid);
					if(err != ErrorCode::ERROR_CODE_NONE)
						return err;
				}
			}
		}
		// 入力レイヤーを複数レイヤーから使用していないか確認する
		for(S32 inputLyaerNum=0; inputLyaerNum<this->layerStructure.inputLayerCount; inputLyaerNum++)
		{
			if(this->GetOutputLayerCount(this->GetInputGUID(inputLyaerNum)) > 1)
			{
				// 出力分割レイヤ－のDLLを取得
				auto pDLL = this->layerDLLManager.GetLayerDLLByGUID(SEPARATE_LAYER_GUID);
				if(pDLL == NULL)
					return ErrorCode::ERROR_CODE_DLL_NOTFOUND;

				// レイヤー構造情報を作成
				auto pLayerStructure = pDLL->CreateLayerStructureSetting();
				if(pLayerStructure == NULL)
					return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
				auto pItem = dynamic_cast<Gravisbell::SettingData::Standard::IItem_Int*>(pLayerStructure->GetItemByID(L"separateCount"));
				if(pItem == NULL)
					return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;

				pItem->SetValue(this->GetOutputLayerCount(this->GetInputGUID(inputLyaerNum)));

				// レイヤーデータを作成
				auto pSubstitutionLayerData = pDLL->CreateLayerData(*pLayerStructure);
				delete pLayerStructure;

				// レイヤーを追加
				ILayerBase* pSubstitutionLayer = NULL;
				err = neuralNetwork.AddTemporaryLayer(pSubstitutionLayerData, &pSubstitutionLayer, i_lpInputDataStruct, i_inputLayerCount, true);
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;

				lpSubstitutionLayer[this->GetInputGUID(inputLyaerNum)] = pSubstitutionLayer->GetGUID();

				// 代替レイヤーの入力を代替先レイヤーに設定
				err = neuralNetwork.AddInputLayerToLayer(pSubstitutionLayer->GetGUID(), this->GetInputGUID(inputLyaerNum));
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;
			}
		}


		// レイヤー間の接続を設定する
		for(auto it_connectInfo : this->lpConnectInfo)
		{
			// 入力レイヤー
			for(auto inputGUID : it_connectInfo.second.lpInputLayerGUID)
			{
				if(lpSubstitutionLayer.count(inputGUID))
				{
					// 代替レイヤーを使用する
					err = neuralNetwork.AddInputLayerToLayer(it_connectInfo.second.guid, lpSubstitutionLayer[inputGUID]);
					if(err != ErrorCode::ERROR_CODE_NONE)
						return err;
				}
				else
				{
					err = neuralNetwork.AddInputLayerToLayer(it_connectInfo.second.guid, inputGUID);
					if(err != ErrorCode::ERROR_CODE_NONE)
						return err;
				}
			}

			// バイパスレイヤー
			for(auto bypassGUID : it_connectInfo.second.lpBypassLayerGUID)
			{
				err = neuralNetwork.AddBypassLayerToLayer(it_connectInfo.second.guid, bypassGUID);
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
		@param	i_pLayerData	追加するレイヤーデータのアドレス.
		@param	i_onFixFlag		レイヤーを固定化するフラグ. */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::AddLayer(const Gravisbell::GUID& i_guid, ILayerData* i_pLayerData, bool i_onFixFlag)
	{
		// レイヤーを検索
		if(this->GetLayerByGUID(i_guid))
			return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;

		// 追加
		this->lpConnectInfo[i_guid] = LayerConnect(i_guid, i_pLayerData, i_onFixFlag);

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
			if(it->second.guid == i_guid)
				break;
		}
		if(it == this->lpConnectInfo.end())
			return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;

		// 削除対象のレイヤーを入力に持つ場合は削除
		for(auto& it_search : this->lpConnectInfo)
		{
			this->EraseInputLayerFromLayer(it_search.second.guid, i_guid);
			this->EraseBypassLayerFromLayer(it_search.second.guid, i_guid);
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
					if(it_search.second.guid != i_guid)
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

		o_guid = it->second.guid;

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 登録されているレイヤーを番号指定で取得する */
	FeedforwardNeuralNetwork_LayerData_Base::LayerConnect* FeedforwardNeuralNetwork_LayerData_Base::GetLayerByNum(U32 i_layerNum)
	{
		if(i_layerNum >= this->lpConnectInfo.size())
			return NULL;

		auto it = this->lpConnectInfo.begin();
		for(U32 i=0; i<i_layerNum; i++)
			it++;

		return &it->second;
	}
	/** 登録されているレイヤーをGUID指定で取得する */
	FeedforwardNeuralNetwork_LayerData_Base::LayerConnect* FeedforwardNeuralNetwork_LayerData_Base::GetLayerByGUID(const Gravisbell::GUID& i_guid)
	{
		auto it = this->lpConnectInfo.find(i_guid);
		if(it == this->lpConnectInfo.end())
			return NULL;

		return &it->second;
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

	/** レイヤーの固定化フラグを取得する */
	bool FeedforwardNeuralNetwork_LayerData_Base::GetLayerFixFlagByGUID(const Gravisbell::GUID& i_guid)
	{
		auto pLayerConnet = this->GetLayerByGUID(i_guid);
		if(pLayerConnet == NULL)
			return false;

		return pLayerConnet->onFixFlag;
	}


	//====================================
	// 入出力レイヤー
	//====================================
	/** 入力信号レイヤー数を取得する */
	U32 FeedforwardNeuralNetwork_LayerData_Base::GetInputCount()
	{
		return this->layerStructure.inputLayerCount;
	}
	/** 入力信号に割り当てられているGUIDを取得する */
	Gravisbell::GUID FeedforwardNeuralNetwork_LayerData_Base::GetInputGUID(U32 i_inputLayerNum)
	{
		if(i_inputLayerNum >= this->lpInputLayerGUID.size())
			return Gravisbell::GUID();
		return this->lpInputLayerGUID[i_inputLayerNum];
	}

	/** 出力信号レイヤーを設定する */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::SetOutputLayerGUID(const Gravisbell::GUID& i_guid)
	{
		auto pLayerConnet = this->GetLayerByGUID(i_guid);
		if(pLayerConnet == NULL)
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		if(pLayerConnet->pLayerData && !pLayerConnet->pLayerData->CheckCanHaveMultOutputLayer())
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
		// 対象レイヤーを入力レイヤーに持つレイヤー数を数える
		U32 outputCount = 0;
		for(auto it : this->lpConnectInfo)
		{
			for(U32 inputNum=0; inputNum<it.second.lpInputLayerGUID.size(); inputNum++)
			{
				if(it.second.lpInputLayerGUID[inputNum] == i_layerGUID)
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


	//===========================
	// オプティマイザー設定
	//===========================
	/** オプティマイザーを変更する */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::ChangeOptimizer(const wchar_t i_optimizerID[])
	{
		for(auto& it : this->lpConnectInfo)
		{
			if(it.second.pLayerData)
				it.second.pLayerData->ChangeOptimizer(i_optimizerID);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** オプティマイザーのハイパーパラメータを変更する */
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value)
	{
		for(auto& it : this->lpConnectInfo)
		{
			if(it.second.pLayerData)
				it.second.pLayerData->SetOptimizerHyperParameter(i_parameterID, i_value);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value)
	{
		for(auto& it : this->lpConnectInfo)
		{
			if(it.second.pLayerData)
				it.second.pLayerData->SetOptimizerHyperParameter(i_parameterID, i_value);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FeedforwardNeuralNetwork_LayerData_Base::SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
	{
		for(auto& it : lpLayerData)
		{
			it.second->SetOptimizerHyperParameter(i_parameterID, i_value);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell
