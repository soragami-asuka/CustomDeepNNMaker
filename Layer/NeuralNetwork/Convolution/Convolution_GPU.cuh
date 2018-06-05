//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̌����������C���[
// GPU�����p
//======================================
#ifndef __CONVOLUTION_GPU_H__
#define __CONVOLUTION_GPU_H__

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


#include"Convolution_DATA.hpp"
#include"Convolution_FUNC.hpp"
#include"Convolution_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class Convolution_GPU : public Convolution_Base
{
private:
	// �f�[�^�{��
	class Convolution_LayerData_GPU& layerData;

	// Get�֐����g���Ə����s�������ނ̂ňꎞ�ۑ��p. PreCalculate�Œl���i�[.
	U32 filterSize;						/**< �t�B���^�T�C�Y */
	U32 inputBufferCount;				/**< ���̓o�b�t�@�� */
	U32 neuronCount;					/**< �j���[������ */
	U32 outputBufferCount;				/**< �o�̓o�b�t�@�� */

	// ���Z�����p�̃o�b�t�@
	cudnnHandle_t cudnnHandle;

	// CUDNN�p�f�[�^�\����`
	cudnnTensorDescriptor_t			inputTensorDesc;			/**< ���̓f�[�^�\�� */
	cudnnTensorDescriptor_t			outputTensorDesc;			/**< �o�̓f�[�^�\�� */
	cudnnTensorDescriptor_t			biasTensorDesc;				/**< �o�C�A�X�f�[�^�\�� */
	cudnnFilterDescriptor_t			filterDesc;					/**< �t�B���^�[�\�� */
	cudnnConvolutionDescriptor_t	convDesc;					/**< ��ݍ��ݐݒ� */
	cudnnConvolutionFwdAlgo_t		useForwardAlgorithm;		/**< �O���`�d���Ɏg�p����A���S���Y���ԍ� */
	cudnnConvolutionBwdDataAlgo_t	useBackwardDataAlgorithm;	/**< ����`�d���̃f�[�^�v�Z�Ɏg�p����A���S���Y���ԍ� */
	cudnnConvolutionBwdFilterAlgo_t	useBackwardFilterAlgorithm;	/**< ����`�d���̃t�B���^�v�Z�Ɏg�p����A���S���Y���ԍ� */

	thrust::device_vector<F32> lpDBias;		/**< �o�C�A�X�̕ω��� */
	thrust::device_vector<F32> lpDNeuron;	/**< �j���[�����̕ω��� */

	Gravisbell::Common::ITemporaryMemoryManager& temporaryMemoryManager;	/**< �ꎞ�o�b�t�@�p�̃������Ǘ��N���X */

public:
	/** �R���X�g���N�^ */
	Convolution_GPU(Gravisbell::GUID guid, class Convolution_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	/** �f�X�g���N�^ */
	virtual ~Convolution_GPU();


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
	Convolution_LayerData_Base& GetLayerData();
	const Convolution_LayerData_Base& GetLayerData()const;


public:
	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLearn();

	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessCalculate();


	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLoop();



	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer);
	ErrorCode Calculate_base(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer, const F32* lpWeight, const F32* lpBias);


public:
	//================================
	// �w�K����
	//================================
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

};



} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif