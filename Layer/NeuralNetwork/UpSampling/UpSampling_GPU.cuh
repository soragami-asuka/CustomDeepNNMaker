//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̌����������C���[
// GPU�����p
//======================================
#ifndef __UpSampling_GPU_H__
#define __UpSampling_GPU_H__

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


#include"UpSampling_DATA.hpp"
#include"UpSampling_FUNC.hpp"
#include"UpSampling_Base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class UpSampling_GPU : public UpSampling_Base
{
private:
	// �f�[�^�{��
	class UpSampling_LayerData_GPU& layerData;

	// ���o�̓o�b�t�@
	thrust::device_vector<F32>			lpOutputBuffer;		/**< �o�̓o�b�t�@ <�o�b�`��><��ݍ��ݐ�> */
	thrust::device_vector<F32>			lpDInputBuffer;		/**< ���͌덷���� <�o�b�`��><���͐M����> */

	thrust::device_vector<F32>			lppDNeuron;	/**< �j���[�����̊w�K��<�j���[������><���͐M����> */
	thrust::device_vector<F32>			lpDBias;	/**< �o�C�A�X�̊w�K��<�j���[������> */

	// Get�֐����g���Ə����s�������ނ̂ňꎞ�ۑ��p. PreCalculate�Œl���i�[.
	U32 inputBufferCount;				/**< ���̓o�b�t�@�� */
	U32 outputBufferCount;				/**< �o�̓o�b�t�@�� */

	// ���Z���̓��̓f�[�^
	CONST_BATCH_BUFFER_POINTER				m_lppInputBuffer_d;			/**< ���Z���̓��̓f�[�^ */
	CONST_BATCH_BUFFER_POINTER				m_lppDOutputBuffer_d;		/**< ���͌덷�v�Z���̏o�͌덷�f�[�^ */

	// ���Z�����p�̃o�b�t�@
	cudnnHandle_t cudnnHandle;

	// CUDNN�p�f�[�^�\����`
	cudnnTensorDescriptor_t			inputTensorDesc;			/**< ���̓f�[�^�\�� */
	cudnnTensorDescriptor_t			upscaleTensorDesc;			/**< ����>�o�͕ϊ��p�̃f�[�^�\�� */
	cudnnTensorDescriptor_t			outputTensorDesc;			/**< �o�̓f�[�^�\�� */

	Gravisbell::Vector3D<S32>		upscaleStride;				/**< ����>�o�͕ϊ��Ɏg�p����ړ��� */

	// �o�͍��� > ���͍����p
	thrust::device_vector<F32>		filter;					/**< �t�B���^�o�b�t�@ */
	cudnnFilterDescriptor_t			filterDesc;				/**< �t�B���^�[�\�� */
	cudnnConvolutionDescriptor_t	backprobConvDesc;		/**< ��ݍ��ݐݒ� */
	cudnnConvolutionFwdAlgo_t		backprobAlgorithm;		/**< ����`�d���Ɏg�p����A���S���Y���ԍ� */
	thrust::device_vector<BYTE>		backprobWorkSpace;		/**< �����p�̃�����.�O���`�d�A����`�d�S�Ăŋ��p����. */

public:
	/** �R���X�g���N�^ */
	UpSampling_GPU(Gravisbell::GUID guid, class UpSampling_LayerData_GPU& i_layerData);
	/** �f�X�g���N�^ */
	virtual ~UpSampling_GPU();


public:
	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 GetLayerKind()const;

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Initialize(void);


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	UpSampling_LayerData_Base& GetLayerData();
	const UpSampling_LayerData_Base& GetLayerData()const;


public:
	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLearn(unsigned int batchSize);

	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessCalculate(unsigned int batchSize);


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& data);
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessCalculateLoop();


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer);

	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const;
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const;

public:
	//================================
	// �w�K����
	//================================
	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER GetDInputBuffer()const;
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const;

};


} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif