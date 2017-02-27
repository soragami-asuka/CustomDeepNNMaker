// IODataLayer.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"
#include "IODataLayer.h"

#include<vector>
#include<list>

#include<rpc.h>
#pragma comment(lib, "Rpcrt4.lib")

namespace CustomDeepNNLibrary
{
	class IODataLayerCPU : public CustomDeepNNLibrary::IIODataLayer
	{
	private:
		GUID guid;	/**< 識別ID */
		CustomDeepNNLibrary::IODataStruct ioDataStruct;	/**< データ構造 */

		std::vector<float*> lpBufferList;
		std::vector<std::vector<float>> lpDInputBuffer;	/**< 誤差差分の保存バッファ */

		unsigned int batchSize;	/**< バッチ処理サイズ */
		const unsigned int* lpBatchDataNoList;	/**< バッチ処理データ番号リスト */

		std::vector<float*> lpBatchDataPointer;			/**< バッチ処理データの配列先頭アドレスリスト */
		std::vector<float*> lpBatchDInputBufferPointer;	/**< バッチ処理入力誤差差分の配列先導アドレスリスト */

	public:
		/** コンストラクタ */
		IODataLayerCPU(GUID guid, CustomDeepNNLibrary::IODataStruct ioDataStruct)
			:	guid				(guid)
			,	ioDataStruct		(ioDataStruct)
			,	lpBatchDataNoList	(NULL)
		{
		}
		/** デストラクタ */
		virtual ~IODataLayerCPU()
		{
			this->ClearData();
		}


		//==============================
		// レイヤー共通系
		//==============================
	public:
		/** レイヤー種別の取得 */
		unsigned int GetLayerKind()const
		{
			return ELayerKind::LAYER_KIND_CPU | ELayerKind::LAYER_KIND_SINGLE_INPUT | ELayerKind::LAYER_KIND_SINGLE_OUTPUT | ELayerKind::LAYER_KIND_DATA;
		}

		/** レイヤー固有のGUIDを取得する */
		ELayerErrorCode GetGUID(GUID& o_guid)const
		{
			o_guid = this->guid;

			return LAYER_ERROR_NONE;
		}

		/** レイヤー種別識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		ELayerErrorCode GetLayerCode(GUID& o_layerCode)const
		{
			// {6E99D406-B931-4DE0-AC3A-48A35E129820}
			o_layerCode = { 0x6e99d406, 0xb931, 0x4de0, { 0xac, 0x3a, 0x48, 0xa3, 0x5e, 0x12, 0x98, 0x20 } };

			return LAYER_ERROR_NONE;
		}

		//==============================
		// データ管理系
		//==============================
	public:
		/** データの構造情報を取得する */
		IODataStruct GetDataStruct()const
		{
			return this->ioDataStruct;
		}

		/** データのバッファサイズを取得する.
			@return データのバッファサイズ.使用するfloat型配列の要素数. */
		unsigned int GetBufferCount()const
		{
			return this->ioDataStruct.ch * this->ioDataStruct.x * this->ioDataStruct.y * this->ioDataStruct.z;
		}

		/** データを追加する.
			@param	lpData	データ一組の配列. GetBufferSize()の戻り値の要素数が必要.
			@return	追加された際のデータ管理番号. 失敗した場合は負の値. */
		ELayerErrorCode AddData(const float lpData[])
		{
			if(lpData == NULL)
				return LAYER_ERROR_COMMON_NULL_REFERENCE;

			// バッファ確保
			float* lpBuffer = new float[this->GetBufferCount()];
			if(lpBuffer == NULL)
				return LAYER_ERROR_COMMON_ALLOCATION_MEMORY;

			// コピー
			memcpy(lpBuffer, lpData, sizeof(float)*this->GetBufferCount());

			// リストに追加
			lpBufferList.push_back(lpBuffer);

			return LAYER_ERROR_NONE;
		}

		/** データ数を取得する */
		unsigned int GetDataCount()const
		{
			return this->lpBufferList.size();
		}
		/** データを番号指定で取得する.
			@param num		取得する番号
			@param o_lpBufferList データの格納先配列. GetBufferSize()の戻り値の要素数が必要.
			@return 成功した場合0 */
		ELayerErrorCode GetDataByNum(unsigned int num, float o_lpBufferList[])const
		{
			if(num >= this->lpBufferList.size())
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			if(o_lpBufferList == NULL)
				return LAYER_ERROR_COMMON_NULL_REFERENCE;

			for(unsigned int i=0; i<this->GetBufferCount(); i++)
			{
				o_lpBufferList[i] = this->lpBufferList[num][i];
			}

			return LAYER_ERROR_NONE;
		}
		/** データを番号指定で消去する */
		ELayerErrorCode EraseDataByNum(unsigned int num)
		{
			if(num >= this->lpBufferList.size())
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

			// 番号の場所まで移動
			auto it = this->lpBufferList.begin();
			for(unsigned int i=0; i<num; i++)
				it++;

			// 削除
			if(*it != NULL)
				delete *it;
			this->lpBufferList.erase(it);

			return LAYER_ERROR_NONE;
		}

		/** データを全消去する.
			@return	成功した場合0 */
		ELayerErrorCode ClearData()
		{
			for(unsigned int i=0; i<lpBufferList.size(); i++)
			{
				if(lpBufferList[i] != NULL)
					delete lpBufferList[i];
			}

			return LAYER_ERROR_NONE;
		}

		/** バッチ処理データ番号リストを設定する.
			設定された値を元にGetDInputBuffer(),GetOutputBuffer()の戻り値が決定する.
			@param i_lpBatchDataNoList	設定するデータ番号リスト. [GetBatchSize()の戻り値]の要素数が必要 */
		ELayerErrorCode SetBatchDataNoList(const unsigned int i_lpBatchDataNoList[])
		{
			this->lpBatchDataNoList = i_lpBatchDataNoList;

			for(unsigned int i=0; i<this->lpBatchDataPointer.size(); i++)
			{
				if(this->lpBatchDataNoList[i] > this->lpBufferList.size())
					return ELayerErrorCode::LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

				this->lpBatchDataPointer[i] = this->lpBufferList[this->lpBatchDataNoList[i]];
			}

			return ELayerErrorCode::LAYER_ERROR_NONE;
		}



		//==============================
		// レイヤー共通系
		//==============================
	public:
		/** 演算前処理を実行する.(学習用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
		ELayerErrorCode PreProcessLearn(unsigned int batchSize)
		{
			// バッチ処理データ配列の初期化
			this->batchSize = batchSize;
			lpBatchDataPointer.resize(batchSize);

			// 誤差差分データ配列の初期化
			this->lpDInputBuffer.resize(batchSize);
			this->lpBatchDInputBufferPointer.resize(batchSize);
			for(unsigned int i=0; i<this->lpDInputBuffer.size(); i++)
			{
				this->lpDInputBuffer[i].resize(this->GetInputBufferCount());
				this->lpBatchDInputBufferPointer[i] = &this->lpDInputBuffer[i][0];
			}

			return ELayerErrorCode::LAYER_ERROR_NONE;
		}

		/** 演算前処理を実行する.(演算用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		ELayerErrorCode PreProcessCalculate(unsigned int batchSize)
		{
			// バッチ処理データ配列の初期化
			this->batchSize = batchSize;
			lpBatchDataPointer.resize(batchSize);

			return ELayerErrorCode::LAYER_ERROR_NONE;
		}

		/** 学習ループの初期化処理.データセットの学習開始前に実行する
			失敗した場合はCalculate以降の処理は実行不可. */
		ELayerErrorCode PreProcessLearnLoop(const INNLayerConfig& config)
		{
			return ELayerErrorCode::LAYER_ERROR_NONE;
		}

		/** バッチサイズを取得する.
			@return 同時に演算を行うバッチのサイズ */
		unsigned int GetBatchSize()const
		{
			return this->batchSize;
		}


		//==============================
		// 入力系
		//==============================
	public:
		/** 学習誤差を計算する.
			@param i_lppInputBuffer	入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要
			直前の計算結果を使用する */
		ELayerErrorCode CalculateLearnError(const float** i_lppInputBuffer)
		{
			const float* lpOutputBuffer = this->GetOutputBuffer();
			if(lpOutputBuffer == NULL)
				return LAYER_ERROR_COMMON_NULL_REFERENCE;

			for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
			{
			}

			unsigned int dataNum=0;
			for(unsigned int layerNum=0; layerNum<this->lppInputFromLayer.size(); layerNum++)
			{
				auto pLayer = this->lppInputFromLayer[layerNum];

				if(pLayer == NULL)
					continue;

				const float* lpInputBuffer = pLayer->GetOutputBuffer();
				if(lpInputBuffer == NULL)
					continue;

				for(unsigned int i=0; i<pLayer->GetOutputBufferCount(); i++)
				{
					this->lpDInputBuffer[dataNum] = lpOutputBuffer[dataNum] - lpInputBuffer[i];

					dataNum++;
				}
			}

			return LAYER_ERROR_NONE;
		}

	public:
		/** 入力バッファ数を取得する. byte数では無くデータの数なので注意 */
		unsigned int GetInputBufferCount()const
		{
			return this->GetBufferCount();
		}

		/** 学習差分を取得する.
			配列の要素数はGetInputBufferCountの戻り値.
			@return	誤差差分配列の先頭ポインタ */
		const float** GetDInputBuffer()const
		{
			return &this->lpDInputBuffer[0];
		}
		/** 学習差分を取得する.
			@param lpDOutputBuffer	学習差分を格納する配列. GetInputBufferCountで取得した値の要素数が必要 */
		ELayerErrorCode GetDInputBuffer(float o_lpDInputBuffer[])const
		{
			if(o_lpDInputBuffer == NULL)
				return LAYER_ERROR_COMMON_NULL_REFERENCE;

			// メモリをコピー
			memcpy(o_lpDInputBuffer, &this->lpDInputBuffer[0], sizeof(float) * this->GetBufferCount());

			return LAYER_ERROR_NONE;
		}

	public:
		/** 入力元レイヤーへのリンクを追加する.
			@param	pLayer	追加する入力元レイヤー
			@return	成功した場合0 */
		ELayerErrorCode AddInputFromLayer(class IOutputLayer* pLayer)
		{
			// 同じ入力レイヤーが存在しない確認する
			for(auto it : this->lppInputFromLayer)
			{
				if(it == pLayer)
					return LAYER_ERROR_ADDLAYER_ALREADY_SAMEID;
			}


			// リストに追加
			this->lppInputFromLayer.push_back(pLayer);

			// 入力元レイヤーに自分を出力先として追加
			pLayer->AddOutputToLayer(this);

			return LAYER_ERROR_NONE;
		}
		/** 入力元レイヤーへのリンクを削除する.
			@param	pLayer	削除する入力元レイヤー
			@return	成功した場合0 */
		ELayerErrorCode EraseInputFromLayer(class IOutputLayer* pLayer)
		{
			// リストから検索して削除
			auto it = this->lppInputFromLayer.begin();
			while(it != this->lppInputFromLayer.end())
			{
				if(*it == pLayer)
				{
					// リストから削除
					this->lppInputFromLayer.erase(it);

					// 削除レイヤーに登録されている自分自身を削除
					pLayer->EraseOutputToLayer(this);

					return LAYER_ERROR_NONE;
				}
				it++;
			}

			return LAYER_ERROR_ERASELAYER_NOTFOUND;
		}

	public:
		/** 入力元レイヤー数を取得する */
		unsigned int GetInputFromLayerCount()const
		{
			return this->lppInputFromLayer.size();
		}
		/** 入力元レイヤーのアドレスを番号指定で取得する.
			@param num	取得するレイヤーの番号.
			@return	成功した場合入力元レイヤーのアドレス.失敗した場合はNULLが返る. */
		IOutputLayer* GetInputFromLayerByNum(unsigned int num)const
		{
			if(num >= this->lppInputFromLayer.size())
				return NULL;

			return this->lppInputFromLayer[num];
		}

		/** 入力元レイヤーが入力バッファのどの位置に居るかを返す.
			※対象入力レイヤーの前にいくつの入力バッファが存在するか.
			　学習差分の使用開始位置としても使用する.
			@return 失敗した場合負の値が返る*/
		int GetInputBufferPositionByLayer(const class IOutputLayer* pLayer)
		{
			unsigned int bufferPos = 0;

			for(unsigned int layerNum=0; layerNum<this->lppInputFromLayer.size(); layerNum++)
			{
				if(this->lppInputFromLayer[layerNum] == pLayer)
					return bufferPos;

				bufferPos += this->lppInputFromLayer[layerNum]->GetOutputBufferCount();
			}

			return -1;
		}

		//==============================
		// 出力系
		//==============================
	public:
		/** 出力データ構造を取得する */
		IODataStruct GetOutputDataStruct()const
		{
			return this->GetDataStruct();
		}

		/** 出力バッファ数を取得する. byte数では無くデータの数なので注意 */
		unsigned int GetOutputBufferCount()const
		{
			return this->GetBufferCount();
		}

		/** 出力データバッファを取得する.
			配列の要素数はGetOutputBufferCountの戻り値.
			@return 出力データ配列の先頭ポインタ */
		const float* GetOutputBuffer()const
		{
			if(this->currentUseNo < 0)
				return NULL;
			if((unsigned int)this->currentUseNo >= this->lpBufferList.size())
				return NULL;

			return this->lpBufferList[this->currentUseNo];
		}
		/** 出力データバッファを取得する.
			@param lpOutputBuffer	出力データ格納先配列. GetOutputBufferCountで取得した値の要素数が必要
			@return 成功した場合0 */
		ELayerErrorCode GetOutputBuffer(float o_lpOutputBuffer[])const
		{
			if(o_lpOutputBuffer == NULL)
				return LAYER_ERROR_COMMON_NULL_REFERENCE;

			if(this->currentUseNo < 0)
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;
			if((unsigned int)this->currentUseNo >= this->lpBufferList.size())
				return LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;


			// データをコピー
			memcpy(o_lpOutputBuffer, this->lpBufferList[this->currentUseNo], sizeof(float)*this->GetBufferCount());

			return LAYER_ERROR_NONE;
		}

	public:
		/** 出力先レイヤーへのリンクを追加する.
			@param	pLayer	追加する出力先レイヤー
			@return	成功した場合0 */
		ELayerErrorCode AddOutputToLayer(class IInputLayer* pLayer)
		{
			// 同じ出力先レイヤーが存在しない確認する
			for(auto it : this->lppOutputToLayer)
			{
				if(it == pLayer)
					return LAYER_ERROR_ADDLAYER_ALREADY_SAMEID;
			}


			// リストに追加
			this->lppOutputToLayer.push_back(pLayer);

			// 出力先レイヤーに自分を入力元として追加
			pLayer->AddInputFromLayer(this);

			return LAYER_ERROR_NONE;
		}
		/** 出力先レイヤーへのリンクを削除する.
			@param	pLayer	削除する出力先レイヤー
			@return	成功した場合0 */
		ELayerErrorCode EraseOutputToLayer(class IInputLayer* pLayer)
		{
			// リストから検索して削除
			auto it = this->lppOutputToLayer.begin();
			while(it != this->lppOutputToLayer.end())
			{
				if(*it == pLayer)
				{
					// リストから削除
					this->lppOutputToLayer.erase(it);

					// 削除レイヤーに登録されている自分自身を削除
					pLayer->EraseInputFromLayer(this);

					return LAYER_ERROR_NONE;
				}
				it++;
			}

			return LAYER_ERROR_ERASELAYER_NOTFOUND;
		}

	public:
		/** 出力先レイヤー数を取得する */
		unsigned int GetOutputToLayerCount()const
		{
			return this->lppOutputToLayer.size();
		}
		/** 出力先レイヤーのアドレスを番号指定で取得する.
			@param num	取得するレイヤーの番号.
			@return	成功した場合出力先レイヤーのアドレス.失敗した場合はNULLが返る. */
		IInputLayer* GetOutputToLayerByNum(unsigned int num)const
		{
			if(num > this->lppOutputToLayer.size())
				return NULL;

			return this->lppOutputToLayer[num];
		}


		//==============================
		// 固有系
		//==============================
		/** 演算前処理を実行する.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		ELayerErrorCode PreCalculate()
		{
			// 入力信号と出力信号のデータ数が一致していることを確認
			for(auto& layer : this->lppInputFromLayer)
			{
				if(layer->GetOutputBufferCount() != this->GetOutputBufferCount())
				{
					return LAYER_ERROR_IO_DISAGREE_INPUT_OUTPUT_COUNT;
				}
			}

			return LAYER_ERROR_NONE;
		}
	};
}

/** 入力信号データレイヤーを作成する.GUIDは自動割り当て.CPU制御
	@param bufferSize	バッファのサイズ.※float型配列の要素数.
	@return	入力信号データレイヤーのアドレス */
extern "C" IODataLayer_API CustomDeepNNLibrary::IIODataLayer* CreateIODataLayerCPU(CustomDeepNNLibrary::IODataStruct ioDataStruct)
{
	UUID uuid;
	::UuidCreate(&uuid);

	return CreateIODataLayerCPUwithGUID(uuid, ioDataStruct);
}
/** 入力信号データレイヤーを作成する.CPU制御
	@param guid			レイヤーのGUID.
	@param bufferSize	バッファのサイズ.※float型配列の要素数.
	@return	入力信号データレイヤーのアドレス */
extern "C" IODataLayer_API CustomDeepNNLibrary::IIODataLayer* CreateIODataLayerCPUwithGUID(GUID guid, CustomDeepNNLibrary::IODataStruct ioDataStruct)
{
	return new CustomDeepNNLibrary::IODataLayerCPU(guid, ioDataStruct);
}