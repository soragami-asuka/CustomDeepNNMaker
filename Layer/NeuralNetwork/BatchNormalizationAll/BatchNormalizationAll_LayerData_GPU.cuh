//======================================
// バッチ正規化のレイヤーデータ
//======================================
#ifndef __BatchNormalizationAll_LAYERDATA_GPU_H__
#define __BatchNormalizationAll_LAYERDATA_GPU_H__

#pragma warning(push)
#pragma warning(disable : 4267)
#pragma warning(disable : 4819)
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)

#include"BatchNormalizationAll_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class BatchNormalizationAll_LayerData_GPU : public BatchNormalizationAll_LayerData_Base
	{
		friend class BatchNormalizationAll_GPU;

	private:
		thrust::device_vector<F32> lpMean;		/**< 各Chの平均 */
		thrust::device_vector<F32> lpVariance;	/**< 各Chの分散 */

		thrust::device_vector<F32>	lpScale;	/**< 各Chのスケール値 */
		thrust::device_vector<F32>	lpBias;		/**< 各Chのバイアス値 */


		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		BatchNormalizationAll_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~BatchNormalizationAll_LayerData_GPU();


		//===========================
		// 初期化
		//===========================
	public:
		/** 初期化. 各ニューロンの値をランダムに初期化
			@return	成功した場合0 */
		ErrorCode Initialize(void);
		/** 初期化. 各ニューロンの値をランダムに初期化
			@param	i_config			設定情報
			@oaram	i_inputDataStruct	入力データ構造情報
			@return	成功した場合0 */
		ErrorCode Initialize(const SettingData::Standard::IData& i_data);
		/** 初期化. バッファからデータを読み込む
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	成功した場合0 */
		ErrorCode InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize );


		//===========================
		// レイヤー保存
		//===========================
	public:
		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const;


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);


		//===========================
		// オプティマイザー設定
		//===========================		
	public:
		/** オプティマイザーを変更する */
		ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[]);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif