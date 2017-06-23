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

#define BLOCK_SIZE	(16)

using namespace Gravisbell;

namespace
{
	/** ベクトルの要素同士の掛け算. */
	__global__ void cuda_func_ConvertImage2Binaryr(const U08* i_lpInputBuffer, F32* o_lpOutputBuffer, U32 i_width, U32 i_height, U32 i_ch, U32 i_bachNum)
	{
		const U32 inputNum = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if(inputNum >= i_width*i_height)	// 分岐するが末尾のwarpだけなので、処理速度に影響はないはず...
			return;
		
		const U32 batchPos = i_bachNum * i_width * i_height * i_ch;

		for(U32 ch=0; ch<i_ch; ch++)
		{
			U32 inputPos  = batchPos +  inputNum * i_ch + ch;
			U32 outputPos = batchPos +  ch * i_width * i_height + inputNum;

			o_lpOutputBuffer[outputPos] = i_lpInputBuffer[inputPos] / 0xFF;
		}
	}
}


namespace Gravisbell {
namespace Layer {
namespace IOData {

	class IODataLayerGPU : public IODataLayerGPU_base
	{
	private:
		std::vector<U08*> lpBufferList;


	public:
		/** コンストラクタ */
		IODataLayerGPU(Gravisbell::GUID guid, U32 i_dataCount, Gravisbell::IODataStruct ioDataStruct)
			:	IODataLayerGPU_base	(guid, ioDataStruct)
		{
			this->lpBufferList.resize(i_dataCount);
		}
		/** デストラクタ */
		virtual ~IODataLayerGPU()
		{
			for(U32 i=0; i<lpBufferList.size(); i++)
			{
				if(lpBufferList[i] != NULL)
					delete lpBufferList[i];
			}
		}



		//==============================
		// データ管理系
		//==============================
	public:
		/** データを追加する.
			@param	lpData	データ一組の配列. GetBufferSize()の戻り値の要素数が必要.
			@return	追加された際のデータ管理番号. 失敗した場合は負の値. */
		Gravisbell::ErrorCode SetData(U32 i_dataNum, const BYTE lpData[], U32 i_lineLength)
		{
			if(lpData == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
			if(i_dataNum >= this->lpBufferList.size())
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			// バッファ確保
			U08* lpBuffer = new U08[this->GetBufferCount()];
			if(lpBuffer == NULL)
				return ErrorCode::ERROR_CODE_COMMON_ALLOCATION_MEMORY;

			// コピー
			for(U32 y=0; y<this->ioDataStruct.y; y++)
			{
				memcpy(&lpBuffer[y*this->ioDataStruct.x*this->ioDataStruct.ch], &lpData[y * i_lineLength], this->ioDataStruct.x*this->ioDataStruct.ch);
			}

			// リストに追加
			if(lpBufferList[i_dataNum])
				delete lpBufferList[i_dataNum];
			lpBufferList[i_dataNum] = lpBuffer;

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
		Gravisbell::ErrorCode GetDataByNum(U32 num, BYTE o_lpBufferList[])const
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

		/** バッチ処理データ番号リストを設定する.
			設定された値を元にGetDInputBuffer(),GetOutputBuffer()の戻り値が決定する.
			@param i_lpBatchDataNoList	設定するデータ番号リスト. [GetBatchSize()の戻り値]の要素数が必要 */
		Gravisbell::ErrorCode SetBatchDataNoList(const U32 i_lpBatchDataNoList[])
		{
			this->lpBatchDataNoList = i_lpBatchDataNoList;

			U32 outputBufferCount = this->GetOutputBufferCount();

			// データを計算用一時バッファにコピー
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				if(this->lpBatchDataNoList[batchNum] > this->lpBufferList.size())
				{
					cudaThreadSynchronize();
					return Gravisbell::ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;
				}

				U32 dataNo = this->lpBatchDataNoList[batchNum];

				cudaMemcpyAsync(
					thrust::raw_pointer_cast(&this->lpTmpImageBuffer[batchNum * outputBufferCount]),
					this->lpBufferList[dataNo],
					sizeof(U32) * outputBufferCount,
					cudaMemcpyHostToDevice);
			}
			cudaThreadSynchronize();

			// 計算用一時バッファに格納したデータをBYTE > F32, y,x,ch > ch,y,x変換する
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				U32 bufferCount = this->ioDataStruct.x * this->ioDataStruct.ch;
				dim3 grid((bufferCount +(BLOCK_SIZE - 1))/BLOCK_SIZE , 1, 1);
				dim3 block(BLOCK_SIZE, 1, 1);

				cuda_func_ConvertImage2Binaryr<<<grid, block>>>(
					thrust::raw_pointer_cast(&this->lpTmpImageBuffer[0]),
					thrust::raw_pointer_cast(&this->lpOutputBuffer[0]),
					ioDataStruct.x, ioDataStruct.y, ioDataStruct.ch,
					batchNum);
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** 入力信号データレイヤーを作成する.GUIDは自動割り当て.CPU制御
		@param bufferSize	バッファのサイズ.※F32型配列の要素数.
		@return	入力信号データレイヤーのアドレス */
	extern IODataLayer_image_API Gravisbell::Layer::IOData::IIODataLayer_image* CreateIODataLayerGPU(Gravisbell::U32 i_dataCount, Gravisbell::U32 i_width, Gravisbell::U32 i_height, Gravisbell::U32 i_ch)
	{
		boost::uuids::uuid uuid = boost::uuids::random_generator()();

		return CreateIODataLayerGPUwithGUID(uuid.data, i_dataCount, i_width, i_height, i_ch);
	}
	/** 入力信号データレイヤーを作成する.CPU制御
		@param guid			レイヤーのGUID.
		@param bufferSize	バッファのサイズ.※F32型配列の要素数.
		@return	入力信号データレイヤーのアドレス */
	extern IODataLayer_image_API Gravisbell::Layer::IOData::IIODataLayer_image* CreateIODataLayerGPUwithGUID(Gravisbell::GUID guid, Gravisbell::U32 i_dataCount, Gravisbell::U32 i_width, Gravisbell::U32 i_height, Gravisbell::U32 i_ch)
	{
		return new Gravisbell::Layer::IOData::IODataLayerGPU(guid, i_dataCount, IODataStruct(i_ch, i_width, i_height, 1));
	}

}	// IOData
}	// Layer
}	// Gravisbell
