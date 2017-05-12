//======================================
// �o�b�`���K���̃��C���[�f�[�^
//======================================
#ifndef __BatchNormalization_LAYERDATA_GPU_H__
#define __BatchNormalization_LAYERDATA_GPU_H__

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

#include"BatchNormalization_LayerData_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class BatchNormalization_LayerData_GPU : public BatchNormalization_LayerData_Base
	{
		friend class BatchNormalization_GPU;

	private:
		thrust::device_vector<F32> lpMean;		/**< �eCh�̕��� */
		thrust::device_vector<F32> lpVariance;	/**< �eCh�̕��U */

		thrust::device_vector<F32>	lpScale;	/**< �eCh�̃X�P�[���l */
		thrust::device_vector<F32>	lpBias;		/**< �eCh�̃o�C�A�X�l */


		//===========================
		// �R���X�g���N�^ / �f�X�g���N�^
		//===========================
	public:
		/** �R���X�g���N�^ */
		BatchNormalization_LayerData_GPU(const Gravisbell::GUID& guid);
		/** �f�X�g���N�^ */
		~BatchNormalization_LayerData_GPU();


		//===========================
		// ������
		//===========================
	public:
		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@return	���������ꍇ0 */
		ErrorCode Initialize(void);
		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@param	i_config			�ݒ���
			@oaram	i_inputDataStruct	���̓f�[�^�\�����
			@return	���������ꍇ0 */
		ErrorCode Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct);
		/** ������. �o�b�t�@����f�[�^��ǂݍ���
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���������ꍇ0 */
		ErrorCode InitializeFromBuffer(const BYTE* i_lpBuffer, U32 i_bufferSize, S32& o_useBufferSize );


		//===========================
		// ���C���[�ۑ�
		//===========================
	public:
		/** ���C���[���o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const;


		//===========================
		// ���C���[�쐬
		//===========================
	public:
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		INNLayer* CreateLayer(const Gravisbell::GUID& guid);
	};

} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif