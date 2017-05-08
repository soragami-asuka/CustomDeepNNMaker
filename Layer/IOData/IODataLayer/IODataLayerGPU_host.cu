//===================================
// 入出力データを管理するクラス
// GPU制御
// hostメモリ確保型
//===================================
#include "stdafx.h"
#include "IODataLayerGPU_base.cuh"


#include<vector>
#include<list>
#include<algorithm>

// UUID関連用
#include<boost/uuid/uuid_generators.hpp>


namespace Gravisbell {
namespace Layer {
namespace IOData {

	class IODataLayerGPU_host : public IODataLayerGPU_base
	{
	private:
		std::vector<F32*> lpBufferList;


	public:
		/** コンストラクタ */
		IODataLayerGPU_host(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
			:	IODataLayerGPU_base	(guid, ioDataStruct)
		{
		}
		/** デストラクタ */
		virtual ~IODataLayerGPU_host()
		{
			this->ClearData();
		}


		//==============================
		// データ管理系
		//==============================
	public:
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
			return (U32)this->lpBufferList.size();
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

			U32 outputBufferCount = this->GetOutputBufferCount();

			// データを出力用バッファにコピー
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				if(this->lpBatchDataNoList[batchNum] > this->lpBufferList.size())
				{
					cudaThreadSynchronize();
					return Gravisbell::ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
				}

				U32 dataNo = this->lpBatchDataNoList[batchNum];

				cudaMemcpyAsync(
					thrust::raw_pointer_cast(&this->lpOutputBuffer[batchNum * outputBufferCount]),
					this->lpBufferList[dataNo],
					sizeof(F32) * outputBufferCount,
					cudaMemcpyHostToDevice);
			}
			cudaThreadSynchronize();

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** 入力信号データレイヤーを作成する.GUIDは自動割り当て.CPU制御
		@param bufferSize	バッファのサイズ.※F32型配列の要素数.
		@return	入力信号データレイヤーのアドレス */
	extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerGPU_host(Gravisbell::IODataStruct ioDataStruct)
	{
		boost::uuids::uuid uuid = boost::uuids::random_generator()();

		return CreateIODataLayerGPUwithGUID_host(uuid.data, ioDataStruct);
	}
	/** 入力信号データレイヤーを作成する.CPU制御
		@param guid			レイヤーのGUID.
		@param bufferSize	バッファのサイズ.※F32型配列の要素数.
		@return	入力信号データレイヤーのアドレス */
	extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerGPUwithGUID_host(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
	{
		return new Gravisbell::Layer::IOData::IODataLayerGPU_host(guid, ioDataStruct);
	}

}	// IOData
}	// Layer
}	// Gravisbell
