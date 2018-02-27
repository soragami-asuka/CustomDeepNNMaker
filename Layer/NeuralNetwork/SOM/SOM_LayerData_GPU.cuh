//======================================
// 全結合ニューラルネットワークのレイヤーデータ
//======================================
#ifndef __SOM_LAYERDATA_CPU_H__
#define __SOM_LAYERDATA_CPU_H__

#include"SOM_LayerData_Base.h"


#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#pragma warning(pop)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class SOM_LayerData_GPU : public SOM_LayerData_Base
	{
		friend class SOM_GPU;

	private:
		// 本体
		thrust::device_vector<F32>		lpUnitData;			/**< 各ユニットの値<ユニット数*入力数> */

		U32								learnTime;			/**< 学習実行回数 */

		//===========================
		// コンストラクタ / デストラクタ
		//===========================
	public:
		/** コンストラクタ */
		SOM_LayerData_GPU(const Gravisbell::GUID& guid);
		/** デストラクタ */
		~SOM_LayerData_GPU();


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
		ErrorCode InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize, S64& o_useBufferSize );


		//===========================
		// レイヤー保存
		//===========================
	public:
		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		S64 WriteToBuffer(BYTE* o_lpBuffer)const;


		//===========================
		// レイヤー作成
		//===========================
	public:
		/** レイヤーを作成する.
			@param guid	新規生成するレイヤーのGUID. */
		ILayerBase* CreateLayer(const Gravisbell::GUID& guid, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);


	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif