//======================================
// フィードフォワードニューラルネットワークの処理レイヤー
// 複数のレイヤーを内包し、処理する
//======================================
#include"stdafx.h"

#include<boost/uuid/uuid_generators.hpp>

#include"FeedforwardNeuralNetwork_Base.h"
#include"FeedforwardNeuralNetwork_LayerData_Base.h"

#include"Layer/NeuralNetwork/INNSingle2SingleLayer.h"
#include"Layer/NeuralNetwork/INNSingle2MultLayer.h"
#include"Layer/NeuralNetwork/INNMult2SingleLayer.h"

#include"LayerConnectInput.h"
#include"LayerConnectOutput.h"
#include"LayerConnectSingle2Single.h"
#include"LayerConnectSingle2Mult.h"
#include"LayerConnectMult2Single.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
	struct BufferInfo
	{
		U32							maxBufferSize;	/**< 最大バッファサイズ */
		std::set<Gravisbell::GUID>	lpUseLayerID;	/**< 使用中のレイヤーID */

		BufferInfo()
			:	maxBufferSize	(0)
			,	lpUseLayerID	()
		{
		}
		BufferInfo(const BufferInfo& info)
			:	maxBufferSize	(info.maxBufferSize)
			,	lpUseLayerID	(lpUseLayerID)
		{
		}
		const BufferInfo& operator=(const BufferInfo& info)
		{
			this->maxBufferSize = info.maxBufferSize;
			this->lpUseLayerID  = info.lpUseLayerID;

			return *this;
		}
	};

	
	//====================================
	// コンストラクタ/デストラクタ
	//====================================
	/** コンストラクタ */
	FeedforwardNeuralNetwork_Base::FeedforwardNeuralNetwork_Base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, const IODataStruct& i_outputDataStruct, Gravisbell::Common::ITemporaryMemoryManager* i_pTemporaryMemoryManager)
		:	layerData			(i_layerData)
		,	guid				(i_guid)			/**< レイヤー識別用のGUID */
		,	outputDataStruct	(i_outputDataStruct)
		,	lppInputLayer		(i_inputLayerCount)		/**< 入力信号の代替レイヤーのアドレス. */
		,	outputLayer			(*this)					/**< 出力信号の代替レイヤーのアドレス. */
		,	pLearnData			(NULL)
		,	pLocalTemporaryMemoryManager	(i_pTemporaryMemoryManager)
		,	temporaryMemoryManager			(*pLocalTemporaryMemoryManager)
		,	lppInputTmpBuffer		(i_inputLayerCount)			/**< 入力バッファ本体 <インプットレイヤー数><バッチ数*入力信号数> */
		,	lppInputBuffer			(i_inputLayerCount)			/**< 入力バッファのアドレス <インプットレイヤー数> */
		,	m_lppInputBuffer		(NULL)	/**< 外部から預かった入力バッファのアドレス(演算デバイス依存) */
		,	m_lppDInputBuffer		(NULL)	/**< 外部から預かった入力誤差バッファのアドレス(演算デバイス依存) */
		,	m_lppDOutputBuffer		(NULL)	/**< 外部から預かった出力誤差バッファのアドレス(演算デバイス依存) */
	{
		for(U32 i=0; i<this->lppInputLayer.size(); i++)
		{
			this->lppInputLayer[i] = new LayerConnectInput(*this, i, i_lpInputDataStruct[i]);
		}
	}
	/** コンストラクタ */
	FeedforwardNeuralNetwork_Base::FeedforwardNeuralNetwork_Base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, const IODataStruct& i_outputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	layerData						(i_layerData)
		,	guid							(i_guid)			/**< レイヤー識別用のGUID */
		,	outputDataStruct				(i_outputDataStruct)
		,	lppInputLayer					(i_inputLayerCount)		/**< 入力信号の代替レイヤーのアドレス. */
		,	outputLayer						(*this)					/**< 出力信号の代替レイヤーのアドレス. */
		,	pLearnData						(NULL)
		,	pLocalTemporaryMemoryManager	(NULL)
		,	temporaryMemoryManager			(i_temporaryMemoryManager)
		,	lppInputTmpBuffer		(i_inputLayerCount)			/**< 入力バッファ本体 <インプットレイヤー数><バッチ数*入力信号数> */
		,	lppInputBuffer			(i_inputLayerCount)			/**< 入力バッファのアドレス <インプットレイヤー数> */
		,	m_lppInputBuffer		(NULL)	/**< 外部から預かった入力バッファのアドレス(演算デバイス依存) */
		,	m_lppDInputBuffer		(NULL)	/**< 外部から預かった入力誤差バッファのアドレス(演算デバイス依存) */
		,	m_lppDOutputBuffer		(NULL)	/**< 外部から預かった出力誤差バッファのアドレス(演算デバイス依存) */
	{
		for(U32 i=0; i<this->lppInputLayer.size(); i++)
		{
			this->lppInputLayer[i] = new LayerConnectInput(*this, i, i_lpInputDataStruct[i]);
		}
	}
	/** デストラクタ */
	FeedforwardNeuralNetwork_Base::~FeedforwardNeuralNetwork_Base()
	{
		// レイヤー処理順序定義の削除
		this->lpCalculateLayerList.clear();

		// 入出力代替レイヤーの削除

		// レイヤー接続情報の削除
		{
			auto it = this->lpLayerInfo.begin();
			while(it != this->lpLayerInfo.end())
			{
				if(it->second)
					delete it->second;
				it = this->lpLayerInfo.erase(it);
			}
		}

		// 一時レイヤーのレイヤーデータを削除
		for(U32 i=0; i<this->lpTemporaryLayerData.size(); i++)
			delete this->lpTemporaryLayerData[i];
		this->lpTemporaryLayerData.clear();

		// 学習データの削除
		if(this->pLearnData)
			delete this->pLearnData;

		// 一時バッファ管理削除
		if(this->pLocalTemporaryMemoryManager != NULL)
			delete this->pLocalTemporaryMemoryManager;

		// 入力信号の代替レイヤーを削除
		for(U32 i=0; i<this->lppInputLayer.size(); i++)
			delete this->lppInputLayer[i];
		this->lppInputLayer.clear();
	}


	//====================================
	// レイヤーの追加
	//====================================
	/** レイヤーを追加する.
		追加したレイヤーの所有権はNeuralNetworkに移るため、メモリの開放処理などは全てINeuralNetwork内で行われる.
			@param	pLayer				追加するレイヤーのアドレス.
			@param	i_onLayerFixFlag	レイヤー固定化フラグ. */
	ErrorCode FeedforwardNeuralNetwork_Base::AddLayer(ILayerBase* pLayer, bool i_onLayerFixFLag)
	{
		// NULLチェック
		if(pLayer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		// 同じレイヤーが登録済みでないことを確認
		if(this->lpLayerInfo.count(pLayer->GetGUID()))
			return ErrorCode::ERROR_CODE_ADDLAYER_ALREADY_SAMEID;

		// レイヤーの入出力種別に応じて分岐
		if((pLayer->GetLayerKind() & Gravisbell::Layer::LAYER_KIND_USETYPE) == LAYER_KIND_NEURALNETWORK)
		{
			INNSingle2SingleLayer*  pSingle2SingleLayer  = dynamic_cast<INNSingle2SingleLayer*>(pLayer);
			INNSingle2MultLayer*    pSingle2MultLayer    = dynamic_cast<INNSingle2MultLayer*>(pLayer);
			INNMult2SingleLayer*    pMult2SingleLayer    = dynamic_cast<INNMult2SingleLayer*>(pLayer);

			if(pSingle2SingleLayer)
			{
				// 単一入力, 単一出力
				this->lpLayerInfo[pLayer->GetGUID()] = new LayerConnectSingle2Single(*this, pLayer, i_onLayerFixFLag);
			}
			else if(pSingle2MultLayer)
			{
				// 単一入力, 複数出力
				this->lpLayerInfo[pLayer->GetGUID()] = new LayerConnectSingle2Mult(*this, pLayer, i_onLayerFixFLag);
			}
			else if(pMult2SingleLayer)
			{
				// 複数入力, 単一出力
				this->lpLayerInfo[pLayer->GetGUID()] = new LayerConnectMult2Single(*this, pLayer, i_onLayerFixFLag);
			}
			else
			{
				// 未対応
				return ErrorCode::ERROR_CODE_ADDLAYER_NOT_COMPATIBLE;
			}
		}
		else
		{
			// 未対応
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_COMPATIBLE;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	
	/** 一時レイヤーを追加する.
		追加したレイヤーデータの所有権はNeuralNetworkに移るため、メモリの開放処理などは全てINeuralNetwork内で行われる.
		@param	i_pLayerData	追加するレイヤーデータ.
		@param	o_player		追加されたレイヤーのアドレス. */
	ErrorCode FeedforwardNeuralNetwork_Base::AddTemporaryLayer(ILayerData* i_pLayerData, ILayerBase** o_pLayer, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, bool i_onLyaerFixFlag)
	{
		if(i_pLayerData == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
		lpTemporaryLayerData.push_back(i_pLayerData);

		ILayerBase* pLayer = i_pLayerData->CreateLayer(boost::uuids::random_generator()().data, i_lpInputDataStruct, i_inputLayerCount, this->temporaryMemoryManager);
		if(pLayer == NULL)
			return ErrorCode::ERROR_CODE_LAYER_CREATE;

		ErrorCode err = this->AddLayer(pLayer, i_onLyaerFixFlag);
		if(err != ErrorCode::ERROR_CODE_NONE)
		{
			delete pLayer;
			return err;
		}
		*o_pLayer = pLayer;

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** レイヤーを削除する.
		@param i_guid	削除するレイヤーのGUID */
	ErrorCode FeedforwardNeuralNetwork_Base::EraseLayer(const Gravisbell::GUID& i_guid)
	{
		auto it = this->lpLayerInfo.find(i_guid);
		if(it == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_ERASELAYER_NOTFOUND;

		if(it->second)
		{
			// 接続解除
			it->second->Disconnect();

			// 本体削除
			delete it->second;
		}
		
		// 領域削除
		this->lpLayerInfo.erase(it);

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** レイヤーを全削除する */
	ErrorCode FeedforwardNeuralNetwork_Base::EraseAllLayer()
	{
		auto it = this->lpLayerInfo.begin();
		while(it != this->lpLayerInfo.end())
		{
			if(it->second)
				delete it->second;
			it = this->lpLayerInfo.erase(it);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 登録されているレイヤー数を取得する */
	U32 FeedforwardNeuralNetwork_Base::GetLayerCount()const
	{
		return (U32)this->lpLayerInfo.size();
	}
	/** レイヤーのGUIDを番号指定で取得する */
	ErrorCode FeedforwardNeuralNetwork_Base::GetLayerGUIDbyNum(U32 i_layerNum, Gravisbell::GUID& o_guid)
	{
		// 範囲チェック
		if(i_layerNum >= this->lpLayerInfo.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		// イテレータを進める
		auto it = this->lpLayerInfo.begin();
		for(U32 i=0; i<i_layerNum; i++)
			it++;

		o_guid = it->first;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//====================================
	// 入出力レイヤー
	//====================================
	/** 入力信号レイヤー数を取得する */
	U32 FeedforwardNeuralNetwork_Base::GetInputCount()
	{
		return this->layerData.GetInputCount();
	}
	/** 入力信号に割り当てられているGUIDを取得する */
	Gravisbell::GUID FeedforwardNeuralNetwork_Base::GetInputGUID(U32 i_inputLayerNum)
	{
		return this->layerData.GetInputGUID(i_inputLayerNum);
	}


	/** 出力信号レイヤーを設定する */
	ErrorCode FeedforwardNeuralNetwork_Base::SetOutputLayerGUID(const Gravisbell::GUID& i_guid)
	{
		// 指定レイヤーが存在することを確認する
		auto it_layer = this->lpLayerInfo.find(i_guid);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 現在の出力レイヤーを削除する
		this->outputLayer.Disconnect();

		// 指定レイヤーを出力レイヤーに接続する
		return this->outputLayer.AddInputLayerToLayer(it_layer->second);
	}


	//====================================
	// レイヤーの接続
	//====================================
	/** レイヤーに入力レイヤーを追加する.
		@param	receiveLayer	入力を受け取るレイヤー
		@param	postLayer		入力を渡す(出力する)レイヤー. */
	ErrorCode FeedforwardNeuralNetwork_Base::AddInputLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer)
	{
		// 受け取り側レイヤーが存在することを確認
		auto it_receive = this->lpLayerInfo.find(receiveLayer);
		if(it_receive == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 入力レイヤーを確認
		for(auto pInputLayer : this->lppInputLayer)
		{
			if(postLayer == pInputLayer->GetGUID())
			{
				return it_receive->second->AddInputLayerToLayer(pInputLayer);
			}
		}

		// 通常レイヤーを確認
		{
			// 出力側レイヤーが存在することを確認
			auto it_post = this->lpLayerInfo.find(postLayer);
			if(it_post == this->lpLayerInfo.end())
				return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

			// 追加処理
			return it_receive->second->AddInputLayerToLayer(it_post->second);
		}
	}
	/** レイヤーにバイパスレイヤーを追加する.
		@param	receiveLayer	入力を受け取るレイヤー
		@param	postLayer		入力を渡す(出力する)レイヤー. */
	ErrorCode FeedforwardNeuralNetwork_Base::AddBypassLayerToLayer(const Gravisbell::GUID& receiveLayer, const Gravisbell::GUID& postLayer)
	{
		// 受け取り側レイヤーが存在することを確認
		auto it_receive = this->lpLayerInfo.find(receiveLayer);
		if(it_receive == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

		// 入力レイヤーを確認
		for(auto pInputLayer : this->lppInputLayer)
		{
			if(postLayer == pInputLayer->GetGUID())
			{
				return it_receive->second->AddBypassLayerToLayer(pInputLayer);
			}
		}

		// 通常レイヤーを確認
		{
			// 出力側レイヤーが存在することを確認
			auto it_post = this->lpLayerInfo.find(postLayer);
			if(it_post == this->lpLayerInfo.end())
				return ErrorCode::ERROR_CODE_ADDLAYER_NOT_EXIST;

			// 追加処理
			return it_receive->second->AddBypassLayerToLayer(it_post->second);
		}
	}

	/** レイヤーの入力レイヤー設定をリセットする.
		@param	layerGUID	リセットするレイヤーのGUID. */
	ErrorCode FeedforwardNeuralNetwork_Base::ResetInputLayer(const Gravisbell::GUID& i_layerGUID)
	{
		// 指定レイヤーが存在することを確認する
		auto it_layer = this->lpLayerInfo.find(i_layerGUID);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		return it_layer->second->ResetInputLayer();
	}
	/** レイヤーのバイパスレイヤー設定をリセットする.
		@param	layerGUID	リセットするレイヤーのGUID. */
	ErrorCode FeedforwardNeuralNetwork_Base::ResetBypassLayer(const Gravisbell::GUID& i_layerGUID)
	{
		// 指定レイヤーが存在することを確認する
		auto it_layer = this->lpLayerInfo.find(i_layerGUID);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		return it_layer->second->ResetBypassLayer();
	}

	/** レイヤーに接続している入力レイヤーの数を取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID. */
	U32 FeedforwardNeuralNetwork_Base::GetInputLayerCount(const Gravisbell::GUID& i_layerGUID)const
	{
		// 指定レイヤーが存在することを確認する
		auto it_layer = this->lpLayerInfo.find(i_layerGUID);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		return it_layer->second->GetInputLayerCount();
	}
	/** レイヤーに接続している入力レイヤーのGUIDを番号指定で取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
		@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
	ErrorCode FeedforwardNeuralNetwork_Base::GetInputLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const
	{
		// 指定レイヤーが存在することを確認する
		auto it_layer = this->lpLayerInfo.find(i_layerGUID);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		// 入力レイヤーの接続情報を取得する
		auto pInputLayer = it_layer->second->GetInputLayerByNum(i_inputNum);
		if(pInputLayer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		// GUIDをコピー
		o_postLayerGUID = pInputLayer->GetGUID();

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** レイヤーに接続しているバイパスレイヤーの数を取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID. */
	U32 FeedforwardNeuralNetwork_Base::GetBypassLayerCount(const Gravisbell::GUID& i_layerGUID)const
	{
		// 指定レイヤーが存在することを確認する
		auto it_layer = this->lpLayerInfo.find(i_layerGUID);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		return it_layer->second->GetBypassLayerCount();
	}
	/** レイヤーに接続しているバイパスレイヤーのGUIDを番号指定で取得する.
		@param	i_layerGUID		接続されているレイヤーのGUID.
		@param	i_inputNum		レイヤーに接続している何番目のレイヤーを取得するかの指定.
		@param	o_postLayerGUID	レイヤーに接続しているレイヤーのGUID格納先. */
	ErrorCode FeedforwardNeuralNetwork_Base::GetBypassLayerGUIDbyNum(const Gravisbell::GUID& i_layerGUID, U32 i_inputNum, Gravisbell::GUID& o_postLayerGUID)const
	{
		// 指定レイヤーが存在することを確認する
		auto it_layer = this->lpLayerInfo.find(i_layerGUID);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		// 入力レイヤーの接続情報を取得する
		auto pInputLayer = it_layer->second->GetBypassLayerByNum(i_inputNum);
		if(pInputLayer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		// GUIDをコピー
		o_postLayerGUID = pInputLayer->GetGUID();

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** レイヤーの接続状態に異常がないかチェックする.
		@param	o_errorLayer	エラーが発生したレイヤーGUID格納先. 
		@return	接続に異常がない場合はNO_ERROR, 異常があった場合は異常内容を返し、対象レイヤーのGUIDをo_errorLayerに格納する. */
	ErrorCode FeedforwardNeuralNetwork_Base::CheckAllConnection(Gravisbell::GUID& o_errorLayer)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}



	//====================================
	// 入力誤差バッファ関連
	//====================================
	/** 各レイヤーが使用する入力誤差バッファを割り当てる */
	ErrorCode FeedforwardNeuralNetwork_Base::AllocateDInputBuffer(void)
	{
		std::map<U32, BufferInfo> lpDInputBufferInfo;	/**< 入力誤差バッファの使用状況<入力誤差バッファのID, 使用中のレイヤーのGUID>  */

		auto it_layer = this->lpCalculateLayerList.rbegin();
		while(it_layer != this->lpCalculateLayerList.rend())
		{
			// 入力レイヤーを走査して入力誤差バッファのIDを割り当てる
			for(U32 inputNum=0; inputNum<(*it_layer)->GetInputLayerCount(); inputNum++)
			{
				auto pInputLayer = (*it_layer)->GetInputLayerByNum(inputNum);
				if(this->GetInputLayerNoByGUID(pInputLayer->GetGUID()) >= 0)
				{
					// 入力レイヤー
					(*it_layer)->SetDInputBufferID(inputNum, (inputNum | NETWORK_DINPUTBUFFER_ID_FLAGBIT) );
				}
				else if(pInputLayer == NULL)
				{
					// 入力が割り当てられていないのでありえないバッファIDを設定する
					(*it_layer)->SetDInputBufferID(inputNum, INVALID_DINPUTBUFFER_ID);
				}
				else
				{
					// 通常レイヤー

					// 未使用の入力誤差バッファを検索
					S32 useDInputBufferID = -1;
					for(auto& it_DInputBuffer : lpDInputBufferInfo)
					{
						if(it_DInputBuffer.second.lpUseLayerID.size() == 0)
						{
							useDInputBufferID = (S32)it_DInputBuffer.first;
							break;
						}
					}
					if(useDInputBufferID < 0)
						useDInputBufferID = (S32)lpDInputBufferInfo.size();
	
					// 入力誤差バッファを使用中に変更し、最大バッファサイズを更新
					lpDInputBufferInfo[(U32)useDInputBufferID].lpUseLayerID.insert(pInputLayer->GetGUID());
					lpDInputBufferInfo[(U32)useDInputBufferID].maxBufferSize = max(lpDInputBufferInfo[(U32)useDInputBufferID].maxBufferSize, pInputLayer->GetOutputDataStruct().GetDataCount());
	
					// 入力誤差バッファのIDを登録
					(*it_layer)->SetDInputBufferID(inputNum, useDInputBufferID);
				}
			}
	
			// 自分が出力誤差として使用している入力誤差バッファを開放
			for(auto& it_DInputBuffer : lpDInputBufferInfo)
			{
				if(it_DInputBuffer.second.lpUseLayerID.count((*it_layer)->GetGUID()) > 0)
				{
					it_DInputBuffer.second.lpUseLayerID.erase((*it_layer)->GetGUID());
				}
			}
	
			it_layer++;
		}
	
		// 入力誤差バッファを確保する
		this->SetDInputBufferCount((U32)lpDInputBufferInfo.size());
		for(U32 dInputBufferNum=0; dInputBufferNum<lpDInputBufferInfo.size(); dInputBufferNum++)
		{
			this->ResizeDInputBuffer(dInputBufferNum, lpDInputBufferInfo[dInputBufferNum].maxBufferSize * this->batchSize);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	//====================================
	// 出力バッファ関連
	//====================================
	/** 各レイヤーが使用する出力バッファを割り当てる */
	ErrorCode FeedforwardNeuralNetwork_Base::AllocateOutputBuffer(void)
	{
		std::map<U32, BufferInfo> lpOutputBufferInfo;	/**< 出力バッファの使用状況<出力バッファのID, 使用中のレイヤーのGUID>  */

		auto it_layer = this->lpCalculateLayerList.begin();
		while(it_layer != this->lpCalculateLayerList.end())
		{
			// 入力レイヤーは出力バッファを持つ必要がないのでスキップ
			// ※入力レイヤーの出力バッファは、ニューラルネットワークの入力バッファ
			if(this->GetInputLayerNoByGUID((*it_layer)->GetGUID()) >= 0)
			{
				it_layer++;
				continue;
			}
			// 出力レイヤーは出力バッファを持つ必要がないのでスキップ
			// ※出力レイヤーの出力バッファは直前のレイヤーの出力バッファを流用
			if((*it_layer)->GetGUID() == this->outputLayer.GetGUID())
			{
				it_layer++;
				continue;
			}

			// 未使用の出力バッファを検索
			S32 useBufferID = -1;
			for(auto& it_DInputBuffer : lpOutputBufferInfo)
			{
				if(it_DInputBuffer.second.lpUseLayerID.size() == 0)
				{
					useBufferID = (S32)it_DInputBuffer.first;
					break;
				}
			}
			if(useBufferID < 0)
				useBufferID = (S32)lpOutputBufferInfo.size();

			// 出力バッファIDを登録する
			(*it_layer)->SetOutputBufferID(useBufferID);

			// 出力バッファのサイズを更新する
			lpOutputBufferInfo[useBufferID].maxBufferSize = max(lpOutputBufferInfo[useBufferID].maxBufferSize, (*it_layer)->GetOutputDataStruct().GetDataCount());

			// 出力レイヤーを走査して使用中レイヤIDを列挙する
			for(U32 outputNum=0; outputNum<(*it_layer)->GetOutputToLayerCount(); outputNum++)
			{
				auto pOutputLayer = (*it_layer)->GetOutputToLayerByNum(outputNum);

				lpOutputBufferInfo[useBufferID].lpUseLayerID.insert(pOutputLayer->GetGUID());
			}

			// 自分が使用しているバッファを開放する
			for(auto& it_buffer : lpOutputBufferInfo)
			{
				if(it_buffer.second.lpUseLayerID.count((*it_layer)->GetGUID()) > 0)
				{
					it_buffer.second.lpUseLayerID.erase((*it_layer)->GetGUID());
				}
			}

			it_layer++;
		}

		// 出力バッファを確保する
		this->SetOutputBufferCount((U32)lpOutputBufferInfo.size());
		for(U32 outputBufferNum=0; outputBufferNum<lpOutputBufferInfo.size(); outputBufferNum++)
		{
			this->ResizeOutputBuffer(outputBufferNum, lpOutputBufferInfo[outputBufferNum].maxBufferSize * this->batchSize);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	//====================================
	// 外部から預かった入出力バッファ関連
	//====================================
	///** 入力バッファを取得する(処理デバイス依存) */
	//CONST_BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_Base::GetInputBuffer()
	//{
	//	return &this->lpInputBuffer[0];
	//}
	/** 入力バッファを取得する(処理デバイス依存) */
	CONST_BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_Base::GetInputBuffer_d(U32 i_inputLayerNum)
	{
		return this->m_lppInputBuffer[i_inputLayerNum];
	}
	/** 入力誤差バッファを取得する(処理デバイス依存) */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_Base::GetDInputBuffer_d(U32 i_inputLayerNum)
	{
		if(this->m_lppDInputBuffer == NULL)
			return NULL;

		return this->m_lppDInputBuffer[i_inputLayerNum];
	}
	/** 出力誤差バッファを取得する */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_Base::GetDOutputBuffer_d()
	{
		return this->m_lppDOutputBuffer;
	}

	/** NNが入力誤差バッファを保持しているかを確認する */
	bool FeedforwardNeuralNetwork_Base::CheckIsHaveDInputBuffer()const
	{
		return (this->m_lppDInputBuffer != NULL);
	}


	//====================================
	// 学習設定
	//====================================
	/** 実行時設定を取得する. */
	const SettingData::Standard::IData* FeedforwardNeuralNetwork_Base::GetRuntimeParameter()const
	{
		return NULL;
	}
	SettingData::Standard::IData* FeedforwardNeuralNetwork_Base::GetRuntimeParameter()
	{
		return NULL;
	}

	/** 学習設定を取得する.
		設定した値はPreProcessLearnLoopを呼び出した際に適用される.
		@param	guid	取得対象レイヤーのGUID. */
	const SettingData::Standard::IData* FeedforwardNeuralNetwork_Base::GetRuntimeParameter(const Gravisbell::GUID& guid)const
	{
		// 指定レイヤーが存在することを確認する
		auto it_layer = this->lpLayerInfo.find(guid);
		if(it_layer == this->lpLayerInfo.end())
			return NULL;

		return it_layer->second->GetRuntimeParameter();
	}
	SettingData::Standard::IData* FeedforwardNeuralNetwork_Base::GetRuntimeParameter(const Gravisbell::GUID& guid)
	{
		return const_cast<SettingData::Standard::IData*>( ((const FeedforwardNeuralNetwork_Base*)this)->GetRuntimeParameter(guid) );
	}
	
	/** 学習設定のアイテムを取得する.
		@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
		@param	i_dataID	設定する値のID. */
	SettingData::Standard::IItemBase* FeedforwardNeuralNetwork_Base::GetRuntimeParameterItem(const Gravisbell::GUID& guid, const wchar_t* i_dataID)
	{
		// 学習設定データを取得
		Gravisbell::SettingData::Standard::IData* pRuntimeParameter = this->GetRuntimeParameter(guid);
		if(pRuntimeParameter == NULL)
			return NULL;

		// 該当IDの設定アイテムを取得
		Gravisbell::SettingData::Standard::IItemBase* pItem = pRuntimeParameter->GetItemByID(i_dataID);
		if(pItem == NULL)
			return NULL;

		return pItem;
	}

	/** 学習設定を設定する.
		設定した値はPreProcessLearnLoopを呼び出した際に適用される.
		int型、float型、enum型が対象.
		@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode FeedforwardNeuralNetwork_Base::SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param)
	{
		for(auto& it : this->lpLayerInfo)
			this->SetRuntimeParameter(it.first, i_dataID, i_param);
		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FeedforwardNeuralNetwork_Base::SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, S32 i_param)
	{
		// 指定レイヤーが存在することを確認する
		auto it_layer = this->lpLayerInfo.find(guid);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		it_layer->second->SetRuntimeParameter(i_dataID, i_param);


		// 該当IDの設定アイテムを取得
		Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetRuntimeParameterItem(guid, i_dataID);
		if(pItem == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 学習設定を設定する.
		設定した値はPreProcessLearnLoopを呼び出した際に適用される.
		int型、float型が対象.
		@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode FeedforwardNeuralNetwork_Base::SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param)
	{
		for(auto& it : this->lpLayerInfo)
			this->SetRuntimeParameter(it.first, i_dataID, i_param);
		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FeedforwardNeuralNetwork_Base::SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, F32 i_param)
	{
		// 指定レイヤーが存在することを確認する
		auto it_layer = this->lpLayerInfo.find(guid);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		it_layer->second->SetRuntimeParameter(i_dataID, i_param);


		// 該当IDの設定アイテムを取得
		Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetRuntimeParameterItem(guid, i_dataID);
		if(pItem == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 学習設定を設定する.
		設定した値はPreProcessLearnLoopを呼び出した際に適用される.
		bool型が対象.
		@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode FeedforwardNeuralNetwork_Base::SetRuntimeParameter(const wchar_t* i_dataID, bool i_param)
	{
		for(auto& it : this->lpLayerInfo)
			this->SetRuntimeParameter(it.first, i_dataID, i_param);
		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FeedforwardNeuralNetwork_Base::SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, bool i_param)
	{
		// 指定レイヤーが存在することを確認する
		auto it_layer = this->lpLayerInfo.find(guid);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		it_layer->second->SetRuntimeParameter(i_dataID, i_param);


		// 該当IDの設定アイテムを取得
		Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetRuntimeParameterItem(guid, i_dataID);
		if(pItem == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 学習設定を設定する.
		設定した値はPreProcessLearnLoopを呼び出した際に適用される.
		string型が対象.
		@param	guid		取得対象レイヤーのGUID.	指定が無い場合は全てのレイヤーに対して実行する.
		@param	i_dataID	設定する値のID.
		@param	i_param		設定する値. */
	ErrorCode FeedforwardNeuralNetwork_Base::SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param)
	{
		for(auto& it : this->lpLayerInfo)
			this->SetRuntimeParameter(it.first, i_dataID, i_param);
		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FeedforwardNeuralNetwork_Base::SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, const wchar_t* i_param)
	{
		// 指定レイヤーが存在することを確認する
		auto it_layer = this->lpLayerInfo.find(guid);
		if(it_layer == this->lpLayerInfo.end())
			return ErrorCode::ERROR_CODE_COMMON_NOT_EXIST;

		it_layer->second->SetRuntimeParameter(i_dataID, i_param);


		// 該当IDの設定アイテムを取得
		Gravisbell::SettingData::Standard::IItemBase* pItem = this->GetRuntimeParameterItem(guid, i_dataID);
		if(pItem == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		return ErrorCode::ERROR_CODE_NONE;
	}



	//===========================
	// レイヤー共通
	//===========================
	/** レイヤー種別の取得.
		ELayerKind の組み合わせ. */
	U32 FeedforwardNeuralNetwork_Base::GetLayerKindBase(void)const
	{
		return Gravisbell::Layer::LAYER_KIND_SINGLE_INPUT | Gravisbell::Layer::LAYER_KIND_SINGLE_OUTPUT | Gravisbell::Layer::LAYER_KIND_NEURALNETWORK;
	}

	/** レイヤー固有のGUIDを取得する */
	Gravisbell::GUID FeedforwardNeuralNetwork_Base::GetGUID(void)const
	{
		return this->guid;
	}

	/** レイヤーの種類識別コードを取得する.
		@param o_layerCode	格納先バッファ
		@return 成功した場合0 */
	Gravisbell::GUID FeedforwardNeuralNetwork_Base::GetLayerCode(void)const
	{
		Gravisbell::GUID guid;
		::GetLayerCode(guid);

		return guid;
	}

	/** バッチサイズを取得する.
		@return 同時に演算を行うバッチのサイズ */
	U32 FeedforwardNeuralNetwork_Base::GetBatchSize()const
	{
		return this->batchSize;
	}

	/** 一時バッファ管理クラスを取得する */
	Common::ITemporaryMemoryManager& FeedforwardNeuralNetwork_Base::GetTemporaryMemoryManager()
	{
		return this->temporaryMemoryManager;
	}

	//================================
	// 初期化処理
	//================================
	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode FeedforwardNeuralNetwork_Base::Initialize(void)
	{
		for(auto& it : this->lpLayerInfo)
		{
			ErrorCode err = it.second->Initialize();
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}
		return ErrorCode::ERROR_CODE_NONE;
	}


	//===========================
	// レイヤー設定
	//===========================
	/** レイヤーの設定情報を取得する */
	const SettingData::Standard::IData* FeedforwardNeuralNetwork_Base::GetLayerStructure()const
	{
		return this->layerData.GetLayerStructure();
	}


	//===========================
	// 入力レイヤー関連
	//===========================
	/** 入力データの数を取得する */
	U32 FeedforwardNeuralNetwork_Base::GetInputDataCount()const
	{
		return (U32)this->lppInputLayer.size();
	}

	/** 入力レイヤー番号をIDから取得する.
		@return	入力レイヤーではない場合は-1,入力レイヤーである場合は番号を0以上で返す */
	S32 FeedforwardNeuralNetwork_Base::GetInputLayerNoByGUID(const Gravisbell::GUID& i_guid)const
	{
		for(S32 no=0; no<this->lppInputLayer.size(); no++)
		{
			if(this->lppInputLayer[no])
			{
				if(this->lppInputLayer[no]->GetGUID() == i_guid)
					return no;
			}
		}
		return -1;
	}

	/** 入力データ構造を取得する.
		@return	入力データ構造 */
	IODataStruct FeedforwardNeuralNetwork_Base::GetInputDataStruct(U32 i_dataNum)const
	{
		if(i_dataNum >= this->lppInputLayer.size())
			return IODataStruct();

		if(this->lppInputLayer[i_dataNum] == NULL)
			return IODataStruct();

		return this->lppInputLayer[i_dataNum]->GetOutputDataStruct();
	}

	/** 入力バッファ数を取得する. */
	U32 FeedforwardNeuralNetwork_Base::GetInputBufferCount(U32 i_dataNum)const
	{
		return this->GetInputDataStruct(i_dataNum).GetDataCount();
	}


	//===========================
	// 出力レイヤー関連
	//===========================
	/** 出力データ構造を取得する */
	IODataStruct FeedforwardNeuralNetwork_Base::GetOutputDataStruct()const
	{
		return this->outputDataStruct;
	}
	IODataStruct FeedforwardNeuralNetwork_Base::GetOutputDataStruct(const GUID& i_layerGUID)const
	{
		auto it_connect = this->lpLayerInfo.find(i_layerGUID);
		if(it_connect == this->lpLayerInfo.end())
			return IODataStruct();

		return it_connect->second->GetOutputDataStruct();
	}

	/** 出力バッファ数を取得する */
	U32 FeedforwardNeuralNetwork_Base::GetOutputBufferCount()const
	{
		return this->GetOutputDataStruct().GetDataCount();
	}


	//================================
	// 演算処理
	//================================
	/** 接続の確立を行う */
	ErrorCode FeedforwardNeuralNetwork_Base::EstablishmentConnection(void)
	{
		ErrorCode err = ErrorCode::ERROR_CODE_NONE;

		// 接続リストをクリア
		this->lpCalculateLayerList.clear();

		// レイヤーのGUIDリストを生成
		std::set<Gravisbell::GUID> lpLayerGUID;	// 全レイヤーリスト
		err = this->outputLayer.CreateLayerList(lpLayerGUID);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// 処理順リストの作成
		std::set<Gravisbell::GUID> lpAddedList;	// 追加済レイヤーリスト
		std::set<ILayerConnect*> lpAddWaitList;	// 追加待機状態リスト
		// 最初の1回目を実行
		err = this->outputLayer.CreateCalculateList(lpLayerGUID, lpCalculateLayerList, lpAddedList, lpAddWaitList);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;
		// 待機状態のレイヤーがある限り実行する
		while(lpAddWaitList.size())
		{
			bool onAddElse = false;	// 一つ以上のレイヤーを追加することができた
			for(auto it : lpAddWaitList)
			{
				bool onHaveNonAddedOutputLayer = false;	// 追加済でないレイヤーを出力対象に保持しているフラグ
				for(U32 outputLayerNum=0; outputLayerNum<it->GetOutputToLayerCount(); outputLayerNum++)
				{
					auto pOutputToLayer = it->GetOutputToLayerByNum(outputLayerNum);

					// レイヤー一覧に含まれていることを確認
					if(lpLayerGUID.count(pOutputToLayer->GetGUID()) == 0)
						continue;

					// 追加待機状態に含まれていないことを確認
					if(lpAddWaitList.count(pOutputToLayer) == 0)
						continue;

					onHaveNonAddedOutputLayer = true;
					break;
				}

				if(onHaveNonAddedOutputLayer == false)
				{
					// 追加待機状態に含まれている出力レイヤーが存在しないため、演算リストに追加可能

					// 追加待機解除
					lpAddWaitList.erase(it);

					// 演算リストに追加
					it->CreateCalculateList(lpLayerGUID, this->lpCalculateLayerList, lpAddedList, lpAddWaitList);

					onAddElse = true;
					break;
				}
			}

			if(!onAddElse)
			{
				// いずれかのレイヤーを追加することができなかった = 再起処理になっているため、エラー
				return ERROR_CODE_COMMON_NOT_COMPATIBLE;
			}
		}


		// 接続の確立
		for(auto& it : this->lpCalculateLayerList)
		{
			err = it->EstablishmentConnection();
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}


		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 演算前処理を実行する.(学習用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
	ErrorCode FeedforwardNeuralNetwork_Base::PreProcessLearn(U32 batchSize)
	{
		ErrorCode err = ErrorCode::ERROR_CODE_NONE;

		// バッチサイズを格納する
		this->batchSize = batchSize;

		// 接続の確立を行う
		err = this->EstablishmentConnection();
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;


		// レイヤーが使用する入力誤差バッファを割り当てる
		err = this->AllocateDInputBuffer();
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// レイヤーが使用する出力バッファを割り当てる
		err = this->AllocateOutputBuffer();
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;


		// 学習の事前処理を実行
		auto it = this->lpCalculateLayerList.begin();
		while(it != this->lpCalculateLayerList.end())
		{
			err = (*it)->PreProcessLearn(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			it++;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode FeedforwardNeuralNetwork_Base::PreProcessCalculate(unsigned int batchSize)
	{
		ErrorCode err = ErrorCode::ERROR_CODE_NONE;

		// バッチサイズを格納する
		this->batchSize = batchSize;

		// 接続の確立を行う
		err = this->EstablishmentConnection();
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// レイヤーが使用する出力バッファを割り当てる
		err = this->AllocateOutputBuffer();
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;


		// 演算の事前処理を実行
		auto it = this->lpCalculateLayerList.begin();
		while(it != this->lpCalculateLayerList.end())
		{
			ErrorCode err = (*it)->PreProcessCalculate(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			it++;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode FeedforwardNeuralNetwork_Base::PreProcessLoop()
	{
		auto it = this->lpCalculateLayerList.begin();
		while(it != this->lpCalculateLayerList.end())
		{
			ErrorCode err = (*it)->PreProcessLoop();
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			it++;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode FeedforwardNeuralNetwork_Base::Calculate_device(CONST_BATCH_BUFFER_POINTER* i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力バッファを保存
		this->m_lppInputBuffer = i_lppInputBuffer;

		// 演算を実行
		auto it = this->lpCalculateLayerList.begin();
		while(it != this->lpCalculateLayerList.end())
		{
			ErrorCode err = (*it)->Calculate();
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			it++;
		}

		// 入力バッファを開放
		this->m_lppInputBuffer = NULL;

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// 学習処理
	//================================
	/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode FeedforwardNeuralNetwork_Base::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER* i_lppInputBuffer, BATCH_BUFFER_POINTER* o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力バッファを保存
		this->m_lppInputBuffer = i_lppInputBuffer;

		// 入力/出力誤差バッファを保存
		this->m_lppDInputBuffer  = o_lppDInputBuffer;
		this->m_lppDOutputBuffer = const_cast<BATCH_BUFFER_POINTER>(i_lppDOutputBuffer);

		// 学習処理を実行
		{
			auto it = this->lpCalculateLayerList.rbegin();
			while(it != this->lpCalculateLayerList.rend())
			{
				ErrorCode err = (*it)->CalculateDInput();
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;

				it++;
			}
		}

		// 入力/出力誤差バッファを解放
		this->m_lppDInputBuffer  = NULL;
		this->m_lppDOutputBuffer = NULL;

		// 入力バッファを開放
		this->m_lppInputBuffer = NULL;

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode FeedforwardNeuralNetwork_Base::Training_device(CONST_BATCH_BUFFER_POINTER* i_lppInputBuffer, BATCH_BUFFER_POINTER* o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力バッファを保存
		this->m_lppInputBuffer = i_lppInputBuffer;

		// 入力/出力誤差バッファを保存
		this->m_lppDInputBuffer  = o_lppDInputBuffer;
		this->m_lppDOutputBuffer = const_cast<BATCH_BUFFER_POINTER>(i_lppDOutputBuffer);

		// 学習処理を実行
		{
			auto it = this->lpCalculateLayerList.rbegin();
			while(it != this->lpCalculateLayerList.rend())
			{
				ErrorCode err = (*it)->Training();
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;

				it++;
			}
		}

		// 入力/出力誤差バッファを解放
		this->m_lppDInputBuffer  = NULL;
		this->m_lppDOutputBuffer = NULL;

		// 入力バッファを開放
		this->m_lppInputBuffer = NULL;

		return ErrorCode::ERROR_CODE_NONE;
	}


}	// NeuralNetwork
}	// Layer
}	// Gravisbell