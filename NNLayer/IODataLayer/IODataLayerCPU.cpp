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

			for(unsigned int batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				if(this->lpBatchDataNoList[batchNum] > this->lpBufferList.size())
					return ELayerErrorCode::LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE;

				this->lpBatchDataPointer[batchNum] = this->lpBufferList[this->lpBatchDataNoList[batchNum]];
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
			this->lpBatchDataPointer.resize(batchSize);

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
			this->lpBatchDataPointer.resize(batchSize);

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
			unsigned int inputBufferCount = this->GetInputBufferCount();

			for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(unsigned int inputNum=0; inputNum<inputBufferCount; inputNum++)
				{
					this->lpDInputBuffer[batchNum][inputNum] = this->lpBatchDataPointer[batchNum][inputNum] - i_lppInputBuffer[batchNum][inputNum];
				}
			}

			return LAYER_ERROR_NONE;
		}

	public:
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		const IODataStruct GetInputDataStruct()const
		{
			return this->GetDataStruct();
		}
		/** 入力データ構造を取得する
			@param	o_inputDataStruct	入力データ構造の格納先
			@return	成功した場合0 */
		ELayerErrorCode GetInputDataStruct(IODataStruct& o_inputDataStruct)const
		{
			o_inputDataStruct = this->GetDataStruct();

			return ELayerErrorCode::LAYER_ERROR_NONE;
		}

		/** 入力バッファ数を取得する. byte数では無くデータの数なので注意 */
		unsigned int GetInputBufferCount()const
		{
			return this->GetBufferCount();
		}

		/** 学習差分を取得する.
			配列の要素数はGetInputBufferCountの戻り値.
			@return	誤差差分配列の先頭ポインタ */
		CONST_BATCH_BUFFER_POINTER GetDInputBuffer()const
		{
			return &this->lpBatchDInputBufferPointer[0];
		}
		/** 学習差分を取得する.
			@param lpDOutputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
		ELayerErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
		{
			if(o_lpDInputBuffer == NULL)
				return LAYER_ERROR_COMMON_NULL_REFERENCE;
			
			const unsigned int batchSize = this->GetBatchSize();
			const unsigned int inputBufferCount = this->GetOutputBufferCount();

			for(unsigned int batchNum=0; batchNum<batchSize; batchNum++)
			{
				memcpy(o_lpDInputBuffer[batchNum], this->lpBatchDataPointer[batchNum], sizeof(float)*inputBufferCount);
			}

			return LAYER_ERROR_NONE;

			return LAYER_ERROR_NONE;
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
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const
		{
			return &this->lpBatchDataPointer[0];
		}
		/** 出力データバッファを取得する.
			@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0 */
		ELayerErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
		{
			if(o_lpOutputBuffer == NULL)
				return LAYER_ERROR_COMMON_NULL_REFERENCE;

			const unsigned int batchSize = this->GetBatchSize();
			const unsigned int outputBufferCount = this->GetOutputBufferCount();

			for(unsigned int batchNum=0; batchNum<batchSize; batchNum++)
			{
				memcpy(o_lpOutputBuffer[batchNum], this->lpBatchDataPointer[batchNum], sizeof(float)*outputBufferCount);
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