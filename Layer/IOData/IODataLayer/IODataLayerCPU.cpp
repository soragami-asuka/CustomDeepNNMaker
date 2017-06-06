// IODataLayer.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"
#include "Library/Layer/IOData/IODataLayer.h"


#include<vector>
#include<list>
#include<algorithm>

// UUID関連用
#include<boost/uuid/uuid_generators.hpp>

namespace Gravisbell {
namespace Layer {
namespace IOData {

	class IODataLayerCPU : public IIODataLayer
	{
	private:
		Gravisbell::GUID guid;	/**< 識別ID */
		Gravisbell::IODataStruct ioDataStruct;	/**< データ構造 */

		std::vector<F32*> lpBufferList;

		U32 batchSize;	/**< バッチ処理サイズ */
		const U32* lpBatchDataNoList;	/**< バッチ処理データ番号リスト */

		std::vector<F32> lpOutputBuffer;	/**< 出力バッファ */
		std::vector<F32> lpDInputBuffer;	/**< 入力誤差バッファ */

		std::vector<F32*> lpBatchDataPointer;			/**< バッチ処理データの配列先頭アドレスリスト */
		std::vector<F32*> lpBatchDInputBufferPointer;	/**< バッチ処理入力誤差差分の配列先導アドレスリスト */

		U32 calcErrorCount;	/**< 誤差計算を実行した回数 */
		std::vector<F32> lpErrorValue_min;	/**< 最小誤差 */
		std::vector<F32> lpErrorValue_max;	/**< 最大誤差 */
		std::vector<F32> lpErrorValue_ave;	/**< 平均誤差 */
		std::vector<F32> lpErrorValue_ave2;	/**< 平均二乗誤差 */
		std::vector<F32> lpErrorValue_crossEntropy;	/**< クロスエントロピー */

		std::vector<S32> lpMaxErrorDataNo;

	public:
		/** コンストラクタ */
		IODataLayerCPU(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
			:	guid				(guid)
			,	ioDataStruct		(ioDataStruct)
			,	lpBatchDataNoList	(NULL)
			,	calcErrorCount		(0)
		{
		}
		/** デストラクタ */
		virtual ~IODataLayerCPU()
		{
			this->ClearData();
		}


		//===========================
		// 初期化
		//===========================
	public:
		/** 初期化. 各ニューロンの値をランダムに初期化
			@return	成功した場合0 */
		ErrorCode Initialize(void)
		{
			return ErrorCode::ERROR_CODE_NONE;
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
		Gravisbell::GUID GetGUID(void)const
		{
			return this->guid;
		}

		/** レイヤー種別識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		Gravisbell::GUID GetLayerCode(void)const
		{
			Gravisbell::GUID layerCode;
			Gravisbell::Layer::IOData::GetLayerCode(layerCode);

			return layerCode;
		}

		/** レイヤーの設定情報を取得する */
		const SettingData::Standard::IData* GetLayerStructure()const
		{
			return NULL;
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
			return this->ioDataStruct.GetDataCount();
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

			// データを出力用バッファにコピー
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				if(this->lpBatchDataNoList[batchNum] > this->lpBufferList.size())
					return Gravisbell::ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

				memcpy(this->lpBatchDataPointer[batchNum], this->lpBufferList[this->lpBatchDataNoList[batchNum]], sizeof(F32)*this->GetBufferCount());
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
			// 通常の演算用の処理を実行
			ErrorCode err = PreProcessCalculate(batchSize);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;

			// 誤差差分データ配列の初期化
			this->lpDInputBuffer.resize(batchSize * this->GetBufferCount());
			this->lpBatchDInputBufferPointer.resize(batchSize);
			for(U32 batchNum=0; batchNum<batchSize; batchNum++)
			{
				this->lpBatchDInputBufferPointer[batchNum] = &this->lpDInputBuffer[batchNum * this->GetBufferCount()];
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** 演算前処理を実行する.(演算用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		Gravisbell::ErrorCode PreProcessCalculate(U32 batchSize)
		{
			// バッチサイズの保存
			this->batchSize = batchSize;

			// バッファの確保とバッチ処理データ配列の初期化
			this->lpOutputBuffer.resize(batchSize * this->GetBufferCount());
			this->lpBatchDataPointer.resize(batchSize);
			for(U32 batchNum=0; batchNum<batchSize; batchNum++)
			{
				this->lpBatchDataPointer[batchNum] = &this->lpOutputBuffer[batchNum * this->GetBufferCount()];
			}

			// 誤差計算処理
			this->lpErrorValue_min.resize(this->GetBufferCount());
			this->lpErrorValue_max.resize(this->GetBufferCount());
			this->lpErrorValue_ave.resize(this->GetBufferCount());
			this->lpErrorValue_ave2.resize(this->GetBufferCount());
			this->lpErrorValue_crossEntropy.resize(this->GetBufferCount());

			this->lpMaxErrorDataNo.resize(this->GetBufferCount());

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}

		/** 学習ループの初期化処理.データセットの学習開始前に実行する
			失敗した場合はCalculate以降の処理は実行不可. */
		Gravisbell::ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& config)
		{
			return this->PreProcessCalculateLoop();
		}
		/** 演算ループの初期化処理.データセットの演算開始前に実行する
			失敗した場合はCalculate以降の処理は実行不可. */
		Gravisbell::ErrorCode PreProcessCalculateLoop()
		{
			this->calcErrorCount = 0;
			this->lpErrorValue_min.assign(this->lpErrorValue_min.size(),  FLT_MAX);
			this->lpErrorValue_max.assign(this->lpErrorValue_max.size(),  0.0f);
			this->lpErrorValue_ave.assign(this->lpErrorValue_ave.size(),  0.0f);
			this->lpErrorValue_ave2.assign(this->lpErrorValue_ave2.size(), 0.0f);
			this->lpErrorValue_crossEntropy.assign(this->lpErrorValue_crossEntropy.size(), 0.0f);

			this->lpMaxErrorDataNo.assign(this->lpMaxErrorDataNo.size(), -1);

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
					F32 output = i_lppInputBuffer[batchNum*inputBufferCount + inputNum];
					F32 teach  = this->lpBatchDataPointer[batchNum][inputNum];

					F32 error = teach - output;
					F32 error_abs = abs(error);

					if(this->lpDInputBuffer.size() > 0)
					{
						this->lpBatchDInputBufferPointer[batchNum][inputNum] = error;
//						this->lpDInputBuffer[batchNum][inputNum] = -(output - teach) / (output * (1.0f - output));
					}

					if(this->lpErrorValue_max[inputNum] < error_abs)
						this->lpMaxErrorDataNo[inputNum] = this->lpBatchDataNoList[batchNum];

					F32 crossEntropy = -(F32)(
						      teach  * log(max(0.0001,  output)) +
						 (1 - teach) * log(max(0.0001,1-output))
						 );

					// 誤差を保存
					this->lpErrorValue_min[inputNum]  = min(this->lpErrorValue_min[inputNum], error_abs);
					this->lpErrorValue_max[inputNum]  = max(this->lpErrorValue_max[inputNum], error_abs);
					this->lpErrorValue_ave[inputNum]  += error_abs;
					this->lpErrorValue_ave2[inputNum] += error_abs * error_abs;
					this->lpErrorValue_crossEntropy[inputNum] += crossEntropy;
				}
				this->calcErrorCount++;
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}


		/** 誤差の値を取得する.
			CalculateLearnError()を1回以上実行していない場合、正常に動作しない.
			@param	o_min	最小誤差.
			@param	o_max	最大誤差.
			@param	o_ave	平均誤差.
			@param	o_ave2	平均二乗誤差. */
		ErrorCode GetCalculateErrorValue(F32& o_min, F32& o_max, F32& o_ave, F32& o_ave2, F32& o_crossEntropy)
		{
			o_min  = FLT_MAX;
			o_max  = 0.0f;
			o_ave  = 0.0f;
			o_ave2 = 0.0f;
			o_crossEntropy = 0.0f;

			for(U32 inputNum=0; inputNum<this->GetBufferCount(); inputNum++)
			{
				o_min   = min(o_min, this->lpErrorValue_min[inputNum]);
				o_max   = max(o_max, this->lpErrorValue_max[inputNum]);
				o_ave  += this->lpErrorValue_ave[inputNum];
				o_ave2 += this->lpErrorValue_ave2[inputNum];
				o_crossEntropy += this->lpErrorValue_crossEntropy[inputNum];
			}

			o_ave  = o_ave / this->calcErrorCount / this->GetBufferCount();
			o_ave2 = (F32)sqrt(o_ave2 / this->calcErrorCount / this->GetBufferCount());
			o_crossEntropy = o_crossEntropy / this->calcErrorCount / this->GetBufferCount();

			//for(U32 inputNum=0; inputNum<this->GetBufferCount(); inputNum++)
			//	printf("%d,", this->lpMaxErrorDataNo[inputNum]);
			//printf("\n");

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** 詳細な誤差の値を取得する.
			各入出力の値毎に誤差を取る.
			CalculateLearnError()を1回以上実行していない場合、正常に動作しない.
			各配列の要素数は[GetBufferCount()]以上である必要がある.
			@param	o_lpMin		最小誤差.
			@param	o_lpMax		最大誤差.
			@param	o_lpAve		平均誤差.
			@param	o_lpAve2	平均二乗誤差. */
		ErrorCode GetCalculateErrorValueDetail(F32 o_lpMin[], F32 o_lpMax[], F32 o_lpAve[], F32 o_lpAve2[])
		{
			for(U32 inputNum=0; inputNum<this->GetBufferCount(); inputNum++)
			{
				o_lpMin[inputNum]   = this->lpErrorValue_min[inputNum];
				o_lpMax[inputNum]   = this->lpErrorValue_max[inputNum];
				o_lpAve[inputNum]  += this->lpErrorValue_ave[inputNum] / this->GetDataCount();
				o_lpAve2[inputNum] += (F32)sqrt(this->lpErrorValue_ave2[inputNum] / this->GetDataCount());
			}
			
			return ErrorCode::ERROR_CODE_NONE;
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
			return &this->lpDInputBuffer[0];
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
				memcpy(&o_lpDInputBuffer[batchNum*inputBufferCount], this->lpBatchDataPointer[batchNum], sizeof(F32)*inputBufferCount);
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
			return &this->lpOutputBuffer[0];
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
				memcpy(&o_lpOutputBuffer[batchNum*outputBufferCount], this->lpBatchDataPointer[batchNum], sizeof(F32)*outputBufferCount);
			}

			return Gravisbell::ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** 入力信号データレイヤーを作成する.GUIDは自動割り当て.CPU制御
		@param bufferSize	バッファのサイズ.※F32型配列の要素数.
		@return	入力信号データレイヤーのアドレス */
	extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerCPU(Gravisbell::IODataStruct ioDataStruct)
	{
		boost::uuids::uuid uuid = boost::uuids::random_generator()();

		return CreateIODataLayerCPUwithGUID(uuid.data, ioDataStruct);
	}
	/** 入力信号データレイヤーを作成する.CPU制御
		@param guid			レイヤーのGUID.
		@param bufferSize	バッファのサイズ.※F32型配列の要素数.
		@return	入力信号データレイヤーのアドレス */
	extern IODataLayer_API Gravisbell::Layer::IOData::IIODataLayer* CreateIODataLayerCPUwithGUID(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct)
	{
		return new Gravisbell::Layer::IOData::IODataLayerCPU(guid, ioDataStruct);
	}

}	// IOData
}	// Layer
}	// Gravisbell


