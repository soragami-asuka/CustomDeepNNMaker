// IODataLayer.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"
#include "IODataLayer.h"

#include<vector>
#include<list>

// UUID関連用
#include<rpc.h>
#pragma comment(lib, "Rpcrt4.lib")

namespace Gravisbell {
namespace Layer {
namespace IOData {

	class IODataLayerCPU : public IIODataLayer
	{
	private:
		GUID guid;	/**< 識別ID */
		Gravisbell::IODataStruct ioDataStruct;	/**< データ構造 */

		std::vector<F32*> lpBufferList;
		std::vector<std::vector<F32>> lpDInputBuffer;	/**< 誤差差分の保存バッファ */

		U32 batchSize;	/**< バッチ処理サイズ */
		const U32* lpBatchDataNoList;	/**< バッチ処理データ番号リスト */

		std::vector<F32*> lpBatchDataPointer;			/**< バッチ処理データの配列先頭アドレスリスト */
		std::vector<F32*> lpBatchDInputBufferPointer;	/**< バッチ処理入力誤差差分の配列先導アドレスリスト */

	public:
		/** コンストラクタ */
		IODataLayerCPU(GUID guid, Gravisbell::IODataStruct ioDataStruct)
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
		U32 GetLayerKind()const
		{
			return ELayerKind::LAYER_KIND_CPU | ELayerKind::LAYER_KIND_SINGLE_INPUT | ELayerKind::LAYER_KIND_SINGLE_OUTPUT | ELayerKind::LAYER_KIND_DATA;
		}

		/** レイヤー固有のGUIDを取得する */
		Gravisbell::ErrorCode GetGUID(GUID& o_guid)const
		{
			o_guid = this->guid;

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** レイヤー種別識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		Gravisbell::ErrorCode GetLayerCode(GUID& o_layerCode)const
		{
			return Gravisbell::Layer::IOData::GetLayerCode(o_layerCode);
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
			@return データのバッファサイズ.使用するF32型配列の要素数. */
		U32 GetBufferCount()const
		{
			return this->ioDataStruct.ch * this->ioDataStruct.x * this->ioDataStruct.y * this->ioDataStruct.z;
		}

		/** データを追加する.
			@param	lpData	データ一組の配列. GetBufferSize()の戻り値の要素数が必要.
			@return	追加された際のデータ管理番号. 失敗した場合は負の値. */
		Gravisbell::ErrorCode AddData(const F32 lpData[])
		{
			if(lpData == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			// バッファ確保
			F32* lpBuffer = new F32[this->GetBufferCount()];
			if(lpBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_ALLOCATION_MEMORY;

			// コピー
			memcpy(lpBuffer, lpData, sizeof(F32)*this->GetBufferCount());

			// リストに追加
			lpBufferList.push_back(lpBuffer);

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** データ数を取得する */
		U32 GetDataCount()const
		{
			return this->lpBufferList.size();
		}
		/** データを番号指定で取得する.
			@param num		取得する番号
			@param o_lpBufferList データの格納先配列. GetBufferSize()の戻り値の要素数が必要.
			@return 成功した場合0 */
		Gravisbell::ErrorCode GetDataByNum(U32 num, F32 o_lpBufferList[])const
		{
			if(num >= this->lpBufferList.size())
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			if(o_lpBufferList == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			for(U32 i=0; i<this->GetBufferCount(); i++)
			{
				o_lpBufferList[i] = this->lpBufferList[num][i];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}
		/** データを番号指定で消去する */
		Gravisbell::ErrorCode EraseDataByNum(U32 num)
		{
			if(num >= this->lpBufferList.size())
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			// 番号の場所まで移動
			auto it = this->lpBufferList.begin();
			for(U32 i=0; i<num; i++)
				it++;

			// 削除
			if(*it != NULL)
				delete *it;
			this->lpBufferList.erase(it);

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** データを全消去する.
			@return	成功した場合0 */
		Gravisbell::ErrorCode ClearData()
		{
			for(U32 i=0; i<lpBufferList.size(); i++)
			{
				if(lpBufferList[i] != NULL)
					delete lpBufferList[i];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** バッチ処理データ番号リストを設定する.
			設定された値を元にGetDInputBuffer(),GetOutputBuffer()の戻り値が決定する.
			@param i_lpBatchDataNoList	設定するデータ番号リスト. [GetBatchSize()の戻り値]の要素数が必要 */
		Gravisbell::ErrorCode SetBatchDataNoList(const U32 i_lpBatchDataNoList[])
		{
			this->lpBatchDataNoList = i_lpBatchDataNoList;

			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				if(this->lpBatchDataNoList[batchNum] > this->lpBufferList.size())
					return Gravisbell::ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

				this->lpBatchDataPointer[batchNum] = this->lpBufferList[this->lpBatchDataNoList[batchNum]];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}



		//==============================
		// レイヤー共通系
		//==============================
	public:
		/** 演算前処理を実行する.(学習用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
		Gravisbell::ErrorCode PreProcessLearn(U32 batchSize)
		{
			// バッチ処理データ配列の初期化
			this->batchSize = batchSize;
			this->lpBatchDataPointer.resize(batchSize);

			// 誤差差分データ配列の初期化
			this->lpDInputBuffer.resize(batchSize);
			this->lpBatchDInputBufferPointer.resize(batchSize);
			for(U32 i=0; i<this->lpDInputBuffer.size(); i++)
			{
				this->lpDInputBuffer[i].resize(this->GetInputBufferCount());
				this->lpBatchDInputBufferPointer[i] = &this->lpDInputBuffer[i][0];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** 演算前処理を実行する.(演算用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		Gravisbell::ErrorCode PreProcessCalculate(U32 batchSize)
		{
			// バッチ処理データ配列の初期化
			this->batchSize = batchSize;
			this->lpBatchDataPointer.resize(batchSize);

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** 学習ループの初期化処理.データセットの学習開始前に実行する
			失敗した場合はCalculate以降の処理は実行不可. */
		Gravisbell::ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& config)
		{
			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** バッチサイズを取得する.
			@return 同時に演算を行うバッチのサイズ */
		U32 GetBatchSize()const
		{
			return this->batchSize;
		}


		//==============================
		// 入力系
		//==============================
	public:
		/** 学習誤差を計算する.
			@param i_lppInputBuffer	入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要 */
		Gravisbell::ErrorCode CalculateLearnError(Gravisbell::CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
		{
			U32 inputBufferCount = this->GetInputBufferCount();

			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(U32 inputNum=0; inputNum<inputBufferCount; inputNum++)
				{
					this->lpDInputBuffer[batchNum][inputNum] = this->lpBatchDataPointer[batchNum][inputNum] - i_lppInputBuffer[batchNum][inputNum];
				}
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

	public:
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		IODataStruct GetInputDataStruct()const
		{
			return this->GetDataStruct();
		}

		/** 入力バッファ数を取得する. byte数では無くデータの数なので注意 */
		U32 GetInputBufferCount()const
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
		Gravisbell::ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
		{
			if(o_lpDInputBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
			
			const U32 batchSize = this->GetBatchSize();
			const U32 inputBufferCount = this->GetOutputBufferCount();

			for(U32 batchNum=0; batchNum<batchSize; batchNum++)
			{
				memcpy(o_lpDInputBuffer[batchNum], this->lpBatchDataPointer[batchNum], sizeof(F32)*inputBufferCount);
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
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
		U32 GetOutputBufferCount()const
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
		Gravisbell::ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
		{
			if(o_lpOutputBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

			const U32 batchSize = this->GetBatchSize();
			const U32 outputBufferCount = this->GetOutputBufferCount();

			for(U32 batchNum=0; batchNum<batchSize; batchNum++)
			{
				memcpy(o_lpOutputBuffer[batchNum], this->lpBatchDataPointer[batchNum], sizeof(F32)*outputBufferCount);
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** 入力信号データレイヤーを作成する.GUIDは自動割り当て.CPU制御
		@param bufferSize	バッファのサイズ.※F32型配列の要素数.
		@return	入力信号データレイヤーのアドレス */
	extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerCPU(Gravisbell::IODataStruct ioDataStruct)
	{
		UUID uuid;
		::UuidCreate(&uuid);

		return CreateIODataLayerCPUwithGUID(uuid, ioDataStruct);
	}
	/** 入力信号データレイヤーを作成する.CPU制御
		@param guid			レイヤーのGUID.
		@param bufferSize	バッファのサイズ.※F32型配列の要素数.
		@return	入力信号データレイヤーのアドレス */
	extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerCPUwithGUID(GUID guid, Gravisbell::IODataStruct ioDataStruct)
	{
		return new Gravisbell::Layer::IOData::IODataLayerCPU(guid, ioDataStruct);
	}

}	// IOData
}	// Layer
}	// Gravisbell
