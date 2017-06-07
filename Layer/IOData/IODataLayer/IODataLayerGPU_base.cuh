//===================================
// 入出力データを管理するクラス
// GPU制御
//===================================
#include "Library/Layer/IOData/IODataLayer.h"


#include<vector>
#include<list>
#include<algorithm>

// UUID関連用
#include<boost/uuid/uuid_generators.hpp>

// CUDA用
#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)


namespace Gravisbell {
namespace Layer {
namespace IOData {

	class IODataLayerGPU_base : public IIODataLayer
	{
	protected:
		Gravisbell::GUID guid;	/**< 識別ID */
		Gravisbell::IODataStruct ioDataStruct;	/**< データ構造 */


		U32 batchSize;	/**< バッチ処理サイズ */
		const U32* lpBatchDataNoList;	/**< バッチ処理データ番号リスト */


		thrust::device_vector<F32>	lpOutputBuffer;	/**< 出力バッファ */
		thrust::device_vector<F32>	lpDInputBuffer;	/**< 入力誤差バッファ */

		U32 calcErrorCount;	/**< 誤差計算を実行した回数 */
		thrust::device_vector<F32>	lpErrorValue_max;	/**< 最大誤差 */
		thrust::device_vector<F32>	lpErrorValue_ave;	/**< 平均誤差 */
		thrust::device_vector<F32>	lpErrorValue_ave2;	/**< 平均二乗誤差 */
		thrust::device_vector<F32>	lpErrorValue_crossEntropy;	/**< クロスエントロピー */

		cublasHandle_t cublasHandle;

	public:
		/** コンストラクタ */
		IODataLayerGPU_base(Gravisbell::GUID guid, Gravisbell::IODataStruct ioDataStruct);
		/** デストラクタ */
		virtual ~IODataLayerGPU_base();


		//===========================
		// 初期化
		//===========================
	public:
		/** 初期化. 各ニューロンの値をランダムに初期化
			@return	成功した場合0 */
		ErrorCode Initialize(void);


		//==============================
		// レイヤー共通系
		//==============================
	public:
		/** レイヤー種別の取得 */
		U32 GetLayerKind()const;

		/** レイヤー固有のGUIDを取得する */
		Gravisbell::GUID GetGUID(void)const;

		/** レイヤー種別識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		Gravisbell::GUID GetLayerCode(void)const;

		/** レイヤーの設定情報を取得する */
		const SettingData::Standard::IData* GetLayerStructure()const;

		//==============================
		// データ管理系
		//==============================
	public:
		/** データの構造情報を取得する */
		IODataStruct GetDataStruct()const;

		/** データのバッファサイズを取得する.
			@return データのバッファサイズ.使用するF32型配列の要素数. */
		U32 GetBufferCount()const;

		/** データを追加する.
			@param	lpData	データ一組の配列. GetBufferSize()の戻り値の要素数が必要.
			@return	追加された際のデータ管理番号. 失敗した場合は負の値. */
		virtual Gravisbell::ErrorCode AddData(const F32 lpData[]) = 0;

		/** データ数を取得する */
		virtual U32 GetDataCount()const = 0;

		/** データを番号指定で取得する.
			@param num		取得する番号
			@param o_lpBufferList データの格納先配列. GetBufferSize()の戻り値の要素数が必要.
			@return 成功した場合0 */
		virtual Gravisbell::ErrorCode GetDataByNum(U32 num, F32 o_lpBufferList[])const = 0;

		/** データを番号指定で消去する */
		virtual Gravisbell::ErrorCode EraseDataByNum(U32 num) = 0;

		/** データを全消去する.
			@return	成功した場合0 */
		virtual Gravisbell::ErrorCode ClearData() = 0;

		/** バッチ処理データ番号リストを設定する.
			設定された値を元にGetDInputBuffer(),GetOutputBuffer()の戻り値が決定する.
			@param i_lpBatchDataNoList	設定するデータ番号リスト. [GetBatchSize()の戻り値]の要素数が必要 */
		virtual Gravisbell::ErrorCode SetBatchDataNoList(const U32 i_lpBatchDataNoList[]) = 0;



		//==============================
		// レイヤー共通系
		//==============================
	public:
		/** 演算前処理を実行する.(学習用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
		Gravisbell::ErrorCode PreProcessLearn(U32 batchSize);

		/** 演算前処理を実行する.(演算用)
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		Gravisbell::ErrorCode PreProcessCalculate(U32 batchSize);

		/** 学習ループの初期化処理.データセットの学習開始前に実行する
			失敗した場合はCalculate以降の処理は実行不可. */
		Gravisbell::ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& config);
		/** 演算ループの初期化処理.データセットの演算開始前に実行する
			失敗した場合はCalculate以降の処理は実行不可. */
		Gravisbell::ErrorCode PreProcessCalculateLoop();

		/** バッチサイズを取得する.
			@return 同時に演算を行うバッチのサイズ */
		U32 GetBatchSize()const;


		//==============================
		// 入力系
		//==============================
	public:
		/** 学習誤差を計算する.
			@param i_lppInputBuffer	入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要 */
		Gravisbell::ErrorCode CalculateLearnError(Gravisbell::CONST_BATCH_BUFFER_POINTER i_lppInputBuffer);


		/** 誤差の値を取得する.
			CalculateLearnError()を1回以上実行していない場合、正常に動作しない.
			@param	o_min	最小誤差.
			@param	o_max	最大誤差.
			@param	o_ave	平均誤差.
			@param	o_ave2	平均二乗誤差. */
		Gravisbell::ErrorCode GetCalculateErrorValue(F32& o_max, F32& o_ave, F32& o_ave2, F32& o_crossEntropy);

		/** 詳細な誤差の値を取得する.
			各入出力の値毎に誤差を取る.
			CalculateLearnError()を1回以上実行していない場合、正常に動作しない.
			各配列の要素数は[GetBufferCount()]以上である必要がある.
			@param	o_lpMin		最小誤差.
			@param	o_lpMax		最大誤差.
			@param	o_lpAve		平均誤差.
			@param	o_lpAve2	平均二乗誤差. */
		Gravisbell::ErrorCode GetCalculateErrorValueDetail(F32 o_lpMax[], F32 o_lpAve[], F32 o_lpAve2[]);


	public:
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		IODataStruct GetInputDataStruct()const;

		/** 入力バッファ数を取得する. byte数では無くデータの数なので注意 */
		U32 GetInputBufferCount()const;

		/** 学習差分を取得する.
			配列の要素数はGetInputBufferCountの戻り値.
			@return	誤差差分配列の先頭ポインタ */
		CONST_BATCH_BUFFER_POINTER GetDInputBuffer()const;
		/** 学習差分を取得する.
			@param lpDOutputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
		Gravisbell::ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const;


		//==============================
		// 出力系
		//==============================
	public:
		/** 出力データ構造を取得する */
		IODataStruct GetOutputDataStruct()const;

		/** 出力バッファ数を取得する. byte数では無くデータの数なので注意 */
		U32 GetOutputBufferCount()const;

		/** 出力データバッファを取得する.
			配列の要素数はGetOutputBufferCountの戻り値.
			@return 出力データ配列の先頭ポインタ */
		CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const;
		/** 出力データバッファを取得する.
			@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0 */
		Gravisbell::ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const;
	};


}	// IOData
}	// Layer
}	// Gravisbell
