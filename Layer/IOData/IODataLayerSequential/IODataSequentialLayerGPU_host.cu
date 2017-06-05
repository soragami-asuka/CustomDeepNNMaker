//===================================
// 入出力データを管理するクラス
// GPU制御
// hostメモリ確保型
//===================================
#include "stdafx.h"
#include "IODataSequentialLayerGPU_base.cuh"


#include<vector>
#include<list>
#include<algorithm>

// UUID関連用
#include<boost/uuid/uuid_generators.hpp>


namespace Gravisbell {
namespace Layer {
namespace IOData {

	class IODataSequentialLayerGPU_host : public IODataSequentialLayerGPU_base
	{
	private:
		std::vector<F32> lpBufferList;
		std::vector<F32*> lpBatchBufferListPointer;


	public:
		/** コンストラクタ */
		IODataSequentialLayerGPU_host(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
			:	IODataSequentialLayerGPU_base	(guid, ioDataStruct)
		{
		}
		/** デストラクタ */
		virtual ~IODataSequentialLayerGPU_host()
		{
		}


		//==============================
		// データ管理系
		//==============================
	public:
		/** データを追加する.
			@param	lpData	データ一組の配列. GetBufferSize()の戻り値の要素数が必要.
			@return	追加された際のデータ管理番号. 失敗した場合は負の値. */
		Gravisbell::ErrorCode SetData(U32 i_dataNo, const F32 lpData[])
		{
			if(lpData == NULL)
				return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;
			if(i_dataNo >= this->lpBatchBufferListPointer.size())
				return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

			// コピー
			memcpy(lpBatchBufferListPointer[i_dataNo], lpData, sizeof(F32)*this->GetBufferCount());

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
				o_lpBufferList[i] = this->lpBatchBufferListPointer[num][i];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

	};

	/** 入力信号データレイヤーを作成する.GUIDは自動割り当て.CPU制御
		@param bufferSize	バッファのサイズ.※F32型配列の要素数.
		@return	入力信号データレイヤーのアドレス */
	extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerGPU_host(Gravisbell::IODataStruct ioDataStruct)
	{
		boost::uuids::uuid uuid = boost::uuids::random_generator()();

		return CreateIODataSequentialLayerGPUwithGUID_host(uuid.data, ioDataStruct);
	}
	/** 入力信号データレイヤーを作成する.CPU制御
		@param guid			レイヤーのGUID.
		@param bufferSize	バッファのサイズ.※F32型配列の要素数.
		@return	入力信号データレイヤーのアドレス */
	extern IODataSequentialLayer_API Gravisbell::Layer::IOData::IIODataSequentialLayer* CreateIODataSequentialLayerGPUwithGUID_host(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
	{
		return new Gravisbell::Layer::IOData::IODataSequentialLayerGPU_host(guid, ioDataStruct);
	}

}	// IOData
}	// Layer
}	// Gravisbell
